use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::Rem;
use std::{fmt, mem};

use empa::buffer;
use empa::buffer::Buffer;
use empa::command::CommandEncoder;
use empa::device::Device;
use empa::query::TimestampQuerySet;
use empa::type_flag::{O, X};

pub struct Measure<L = &'static str> {
    label: L,
    duration: i64,
}

impl<L> Measure<L> {
    pub fn label(&self) -> &L {
        &self.label
    }

    pub fn duration(&self) -> i64 {
        self.duration
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MeasureError<L = &'static str> {
    StartNotFound(L),
    EndNotFound(L),
}

impl<L> fmt::Display for MeasureError<L>
where
    L: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeasureError::StartNotFound(label) => write!(f, "start label not found: {}", label),
            MeasureError::EndNotFound(label) => write!(f, "end label not found: {}", label),
        }
    }
}

impl<L> Error for MeasureError<L> where L: Debug + Display {}

pub struct ResolvedLabelledTimestamps<L = &'static str> {
    inner: HashMap<L, u64>,
}

impl<L> ResolvedLabelledTimestamps<L>
where
    L: Eq + Hash + Clone,
{
    pub fn new() -> Self {
        ResolvedLabelledTimestamps {
            inner: Default::default(),
        }
    }

    pub fn get(&self, key: &L) -> Option<u64> {
        self.inner.get(key).copied()
    }

    pub fn measure(&self, timing: Timing<L>) -> Result<Measure<L>, MeasureError<L>> {
        let start = self
            .inner
            .get(&timing.start_label)
            .copied()
            .ok_or(MeasureError::StartNotFound(timing.start_label))?;
        let end = self
            .inner
            .get(&timing.end_label)
            .copied()
            .ok_or(MeasureError::EndNotFound(timing.end_label))?;

        let duration = end as i64 - start as i64;

        Ok(Measure {
            label: timing.name,
            duration,
        })
    }
}

pub struct LabelledTimestampRecorder<L = &'static str> {
    device: Device,
    query_set_capacity: usize,
    query_sets: Vec<TimestampQuerySet>,
    timings_resolve: Buffer<[u64], buffer::Usages<X, O, O, O, O, O, O, X, O, O>>,
    timings_readback: Buffer<[u64], buffer::Usages<O, O, O, O, O, O, X, O, O, X>>,
    mapping: HashMap<L, usize>,
    index_use: Vec<usize>,
}

impl<L> LabelledTimestampRecorder<L>
where
    L: Eq + Hash + Clone,
{
    pub fn new(device: Device) -> Self {
        let query_set_capacity = 256;

        let query_sets = vec![device.create_timestamp_query_set(query_set_capacity)];
        let timings_resolve = device.create_slice_buffer_zeroed(
            query_set_capacity as usize,
            buffer::Usages::query_resolve().and_copy_src(),
        );
        let timings_readback = device.create_slice_buffer_zeroed(
            query_set_capacity as usize,
            buffer::Usages::copy_dst().and_map_read(),
        );
        let mapping = HashMap::new();
        let index_use = vec![0; 4];

        LabelledTimestampRecorder {
            device,
            query_set_capacity,
            query_sets,
            timings_resolve,
            timings_readback,
            mapping,
            index_use,
        }
    }

    fn timestamp_capacity(&self) -> usize {
        self.query_set_capacity as usize * self.query_sets.len()
    }

    pub fn mark(&mut self, encoder: CommandEncoder, key: L) -> CommandEncoder {
        let LabelledTimestampRecorder {
            mapping, index_use, ..
        } = self;

        let index = *mapping.entry(key).or_insert_with(|| {
            let mut index = None;

            for (i, bitset) in index_use.iter_mut().enumerate() {
                let pos = bitset.trailing_ones() as usize;

                if pos < 32 {
                    index = Some(i * 32 + pos);

                    *bitset |= 1 << pos;

                    break;
                }
            }

            index.unwrap_or_else(|| {
                let i = index_use.len();

                index_use.push(0x1);

                i * 32
            })
        });

        let (query_set, entry) = if index < self.timestamp_capacity() {
            let query_set = index / self.query_set_capacity;
            let entry = index.rem(self.query_set_capacity);

            (query_set as usize, entry)
        } else {
            let query_set = self.query_sets.len();

            let new_query_set = self
                .device
                .create_timestamp_query_set(self.query_set_capacity);

            self.query_sets.push(new_query_set);

            (query_set, 0)
        };

        encoder.write_timestamp(&self.query_sets[query_set], entry)
    }

    pub async fn resolve_into(&mut self, resolved_timings: &mut ResolvedLabelledTimestamps<L>) {
        let mut encoder = self.device.create_command_encoder();

        let required_buffer_capacity = self.query_set_capacity as usize * self.query_sets.len();

        if self.timings_resolve.len() < required_buffer_capacity {
            self.timings_resolve = self.device.create_slice_buffer_zeroed(
                required_buffer_capacity,
                buffer::Usages::query_resolve().and_copy_src(),
            );
            self.timings_readback = self.device.create_slice_buffer_zeroed(
                required_buffer_capacity,
                buffer::Usages::copy_dst().and_map_read(),
            );
        }

        for i in 0..self.query_sets.len() {
            let start = i * self.query_set_capacity as usize;
            let end = start + self.query_set_capacity as usize;

            encoder = encoder.resolve_timestamp_query_set(
                &self.query_sets[i],
                0,
                self.timings_resolve.get(start..end).unwrap(),
            );
        }

        encoder = encoder
            .copy_buffer_to_buffer_slice(self.timings_resolve.view(), self.timings_readback.view());

        self.device.queue().submit(encoder.finish());

        self.timings_readback.map_read().await.unwrap();

        let mapped = self.timings_readback.mapped();

        resolved_timings.inner.clear();
        resolved_timings.inner.reserve(self.mapping.len());

        for (key, index) in self.mapping.iter() {
            resolved_timings
                .inner
                .insert(key.clone(), mapped[*index as usize]);
        }

        mem::drop(mapped);
        self.timings_readback.unmap();
    }
}

pub struct Timing<L = &'static str> {
    pub name: L,
    pub start_label: L,
    pub end_label: L,
}

pub struct AveragedTimings<L = &'static str> {
    timings: Vec<Timing<L>>,
    times: Vec<i64>,
    window_size: usize,
    cursor: usize,
}

impl<L> AveragedTimings<L>
where
    L: Debug + Eq + Hash + Clone,
{
    pub fn new(window_size: usize, timings: Vec<Timing<L>>) -> Self {
        let timings_len = window_size * timings.len();
        let times = vec![0; timings_len];

        AveragedTimings {
            timings,
            times,
            window_size,
            cursor: 0,
        }
    }

    pub fn update(&mut self, resolved_timestamps: &ResolvedLabelledTimestamps<L>) {
        for (i, timing) in self.timings.iter().enumerate() {
            let start = resolved_timestamps
                .get(&timing.start_label)
                .unwrap_or_else(|| panic!("missing timestamp labelled `{:?}`", timing.start_label));
            let end = resolved_timestamps
                .get(&timing.end_label)
                .unwrap_or_else(|| panic!("missing timestamp labelled `{:?}`", timing.end_label));

            let dif = end as i64 - start as i64;
            let index = i * self.window_size + self.cursor;

            self.times[index] = dif;
        }

        self.cursor += 1;

        if self.cursor == self.window_size {
            self.cursor = 0;
        }
    }

    pub fn iter(&self) -> AveragedTimingsIter<L> {
        AveragedTimingsIter {
            timings: self,
            current: 0,
        }
    }
}

pub struct AveragedTimingsIter<'a, L> {
    timings: &'a AveragedTimings<L>,
    current: usize,
}

impl<L> Iterator for AveragedTimingsIter<'_, L>
where
    L: Clone,
{
    type Item = (L, i64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.timings.timings.len() {
            let name = self.timings.timings[self.current].name.clone();

            let timings_offset = self.current * self.timings.window_size;
            let mut sum = 0;

            for i in 0..self.timings.window_size {
                sum += self.timings.times[timings_offset + i];
            }

            let average = sum / self.timings.window_size as i64;

            self.current += 1;

            Some((name, average))
        } else {
            None
        }
    }
}
