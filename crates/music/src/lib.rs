//! Castle Music DSL - A domain-specific language for music composition
//!
//! Inspired by castle_audio_core, this crate provides:
//! - Note constants like `c4q` (C4 quarter), `gs5h` (G#5 half), `rq` (quarter rest)
//! - Serial composition with `ser()` for melodies
//! - Parallel composition with `par()` for harmony/chords
//! - Advanced features: parmin, forkseq, forkpar, transpose, key signatures
//! - Streaming iterator architecture for infinite/looping compositions
//! - MIDI file generation for playback with SoundFonts
//! - BasicSynth for audio synthesis from MIDI events
//!
//! # Example
//! ```ignore
//! use music::prelude::*;
//!
//! let melody = ser![
//!     c4q, d4q, e4q, f4q,
//!     par![c4w, e4w, g4w],  // C major chord
//! ];
//!
//! // Stream events for real-time playback
//! for event in melody.iter_events(120.0) {
//!     // Process each MIDI event as it's generated
//! }
//!
//! // Or generate complete MIDI file
//! let midi_bytes = compose_to_midi(&melody, 120.0);
//!
//! // Or render to audio samples
//! let events = note_to_events(&melody, 120.0);
//! let samples = synth::render_to_samples(&events, 44100);
//! ```

// Synthesizer module
pub mod synth;

// Sound effects module
pub mod sfx;

// =============================================================================
// CORE TYPES
// =============================================================================

/// Time unit for specifying durations in envelopes
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TimeUnit {
    /// Duration in seconds
    Seconds,
    /// Duration in milliseconds
    Milliseconds,
    /// Duration as percentage of the note's total duration (0.0 to 100.0)
    Percent,
}

/// Duration representation for envelope points
#[derive(Clone, Debug, PartialEq)]
pub enum EnvelopeDuration {
    /// Duration as a note value (w, h, q, e, etc.)
    Note(Duration),
    /// Duration as time (seconds, milliseconds, or percent)
    Time(f32, TimeUnit),
}

impl EnvelopeDuration {
    /// Convert envelope duration to seconds given tempo and time parameters
    pub fn to_seconds(&self, tempo: f32, time_note: f32, total_duration: Option<f32>) -> f32 {
        match self {
            EnvelopeDuration::Note(duration) => {
                let seconds_per_beat = 60.0 / tempo;
                let beats_per_whole_note = 4.0;
                let beats_per_quarter = 4.0 / time_note;
                duration.value() * beats_per_whole_note * beats_per_quarter * seconds_per_beat
            }
            EnvelopeDuration::Time(value, unit) => match unit {
                TimeUnit::Seconds => *value,
                TimeUnit::Milliseconds => *value / 1000.0,
                TimeUnit::Percent => {
                    if let Some(total) = total_duration {
                        total * (*value / 100.0)
                    } else {
                        0.0
                    }
                }
            },
        }
    }
}

/// A point in an envelope curve with a value and duration
#[derive(Clone, Debug, PartialEq)]
pub struct EnvelopePoint {
    pub value: f32,
    pub duration: EnvelopeDuration,
    /// If true, value is relative to previous point (e.g., +0.5 or -0.2)
    pub relative: bool,
}

impl EnvelopePoint {
    /// Create an envelope point with a note duration (absolute value)
    pub fn new_note(value: f32, duration: Duration) -> Self {
        EnvelopePoint {
            value,
            duration: EnvelopeDuration::Note(duration),
            relative: false,
        }
    }

    /// Create an envelope point with a note duration (relative value)
    pub fn new_note_relative(value: f32, duration: Duration) -> Self {
        EnvelopePoint {
            value,
            duration: EnvelopeDuration::Note(duration),
            relative: true,
        }
    }

    /// Create an envelope point with duration in seconds (absolute value)
    pub fn new_seconds(value: f32, seconds: f32) -> Self {
        EnvelopePoint {
            value,
            duration: EnvelopeDuration::Time(seconds, TimeUnit::Seconds),
            relative: false,
        }
    }

    /// Create an envelope point with duration in seconds (relative value)
    pub fn new_seconds_relative(value: f32, seconds: f32) -> Self {
        EnvelopePoint {
            value,
            duration: EnvelopeDuration::Time(seconds, TimeUnit::Seconds),
            relative: true,
        }
    }

    /// Create an envelope point with duration in milliseconds (absolute value)
    pub fn new_ms(value: f32, ms: f32) -> Self {
        EnvelopePoint {
            value,
            duration: EnvelopeDuration::Time(ms, TimeUnit::Milliseconds),
            relative: false,
        }
    }

    /// Create an envelope point with duration in milliseconds (relative value)
    pub fn new_ms_relative(value: f32, ms: f32) -> Self {
        EnvelopePoint {
            value,
            duration: EnvelopeDuration::Time(ms, TimeUnit::Milliseconds),
            relative: true,
        }
    }

    /// Create an envelope point with duration as percentage (absolute value)
    pub fn new_percent(value: f32, percent: f32) -> Self {
        EnvelopePoint {
            value,
            duration: EnvelopeDuration::Time(percent, TimeUnit::Percent),
            relative: false,
        }
    }

    /// Create an envelope point with duration as percentage (relative value)
    pub fn new_percent_relative(value: f32, percent: f32) -> Self {
        EnvelopePoint {
            value,
            duration: EnvelopeDuration::Time(percent, TimeUnit::Percent),
            relative: true,
        }
    }
}

/// Defines the interpolation method used between envelope points
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InterpolationType {
    /// Linear interpolation between points (default)
    Linear,
    /// Cosine interpolation for smoother transitions
    Cosine,
    /// Exponential interpolation for sharper or more gradual transitions
    Exponential,
    /// Cubic interpolation for smoother, more natural transitions
    Cubic,
}

/// An envelope defines a series of values over time for modulating parameters
#[derive(Clone, Debug, PartialEq)]
pub struct Envelope {
    /// The parameter this envelope targets (e.g., "volume", "pitch")
    pub target: String,
    /// The points defining the envelope curve
    pub points: Vec<EnvelopePoint>,
    /// The interpolation method used between points
    pub interpolation: InterpolationType,
}

impl Envelope {
    /// Create a new envelope with linear interpolation
    pub fn with_points(target: &str, points: Vec<EnvelopePoint>) -> Self {
        Envelope {
            target: target.to_string(),
            points,
            interpolation: InterpolationType::Linear,
        }
    }

    /// Create a new envelope with the specified interpolation type
    pub fn with_interpolation(
        target: &str,
        points: Vec<EnvelopePoint>,
        interpolation: InterpolationType,
    ) -> Self {
        Envelope {
            target: target.to_string(),
            points,
            interpolation,
        }
    }

    /// Evaluate the envelope at a specific time point
    pub fn evaluate(&self, time: f32, tempo: f32, time_note: f32, total_duration: Option<f32>) -> f32 {
        if self.points.is_empty() {
            return 0.0;
        }

        if self.points.len() == 1 {
            let first = &self.points[0];
            // First point: relative means relative to 0
            return if first.relative { first.value } else { first.value };
        }

        // Calculate absolute time and resolved value for each point
        // Relative values are accumulated from previous points
        let mut point_times: Vec<(f32, f32)> = Vec::with_capacity(self.points.len());
        let mut cumulative_time = 0.0;
        let mut current_value = 0.0; // Track current absolute value for relative calculations

        for (idx, point) in self.points.iter().enumerate() {
            let point_time = if idx == 0 {
                0.0
            } else {
                cumulative_time + point.duration.to_seconds(tempo, time_note, total_duration)
            };

            // Resolve the value: relative adds to previous, absolute replaces
            let resolved_value = if point.relative {
                current_value + point.value
            } else {
                point.value
            };
            current_value = resolved_value;

            point_times.push((point_time, resolved_value));
            cumulative_time = point_time;
        }

        // If time is before the first point, return the first value
        if time <= point_times[0].0 + 1e-6 {
            return point_times[0].1;
        }

        // Find surrounding points and interpolate
        for idx in 0..(point_times.len() - 1) {
            let (prev_time, prev_val) = point_times[idx];
            let (next_time, next_val) = point_times[idx + 1];

            if time >= prev_time - 1e-6 && time <= next_time + 1e-6 {
                let segment_duration = next_time - prev_time;

                if segment_duration < 1e-6 {
                    return if (time - next_time).abs() < (time - prev_time).abs() {
                        next_val
                    } else {
                        prev_val
                    };
                }

                // Calculate interpolation factor
                let interp_t = ((time - prev_time) / segment_duration).clamp(0.0, 1.0);

                // Apply interpolation based on type
                return match self.interpolation {
                    InterpolationType::Linear => {
                        prev_val + interp_t * (next_val - prev_val)
                    }
                    InterpolationType::Cosine => {
                        let cosine_t = (1.0 - f32::cos(interp_t * std::f32::consts::PI)) * 0.5;
                        prev_val + cosine_t * (next_val - prev_val)
                    }
                    InterpolationType::Exponential => {
                        // Fall back to linear for non-positive values
                        if prev_val <= 0.0 || next_val <= 0.0 {
                            prev_val + interp_t * (next_val - prev_val)
                        } else {
                            let exp_t = interp_t * interp_t;
                            prev_val * f32::powf(next_val / prev_val, exp_t)
                        }
                    }
                    InterpolationType::Cubic => {
                        // Smoothstep interpolation
                        let cubic_t = interp_t * interp_t * (3.0 - 2.0 * interp_t);
                        prev_val + cubic_t * (next_val - prev_val)
                    }
                };
            }
        }

        // If time is after the last point, return the last value
        point_times.last().unwrap().1
    }

    /// Get the total duration of the envelope in seconds
    pub fn total_duration(&self, tempo: f32, time_note: f32) -> f32 {
        let mut total = 0.0;
        for (idx, point) in self.points.iter().enumerate() {
            if idx > 0 {
                total += point.duration.to_seconds(tempo, time_note, None);
            }
        }
        total
    }
}

/// Duration as fraction of a whole note
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Duration {
    // Base durations
    Whole,              // 1.0
    Half,               // 0.5
    Quarter,            // 0.25
    Eighth,             // 0.125
    Sixteenth,          // 0.0625
    ThirtySecond,       // 0.03125
    SixtyFourth,        // 0.015625
    OneTwentyEighth,    // 0.0078125
    // Single dotted (1.5x)
    DottedWhole,        // 1.5
    DottedHalf,         // 0.75
    DottedQuarter,      // 0.375
    DottedEighth,       // 0.1875
    DottedSixteenth,    // 0.09375
    DottedThirtySecond, // 0.046875
    DottedSixtyFourth,  // 0.0234375
    DottedOneTwentyEighth, // 0.01171875
    // Double dotted (1.75x)
    DoubleDottedWhole,  // 1.75
    DoubleDottedHalf,   // 0.875
    DoubleDottedQuarter, // 0.4375
    DoubleDottedEighth, // 0.21875
    DoubleDottedSixteenth, // 0.109375
    DoubleDottedThirtySecond, // 0.0546875
    DoubleDottedSixtyFourth, // 0.02734375
    DoubleDottedOneTwentyEighth, // 0.013671875
    // Triple dotted (1.875x)
    TripleDottedWhole,  // 1.875
    TripleDottedHalf,   // 0.9375
    TripleDottedQuarter, // 0.46875
    TripleDottedEighth, // 0.234375
    TripleDottedSixteenth, // 0.1171875
    TripleDottedThirtySecond, // 0.05859375
    TripleDottedSixtyFourth, // 0.029296875
    TripleDottedOneTwentyEighth, // 0.0146484375
}

impl Duration {
    /// Get duration as fraction of whole note
    pub fn value(&self) -> f32 {
        match self {
            Duration::Whole => 1.0,
            Duration::Half => 0.5,
            Duration::Quarter => 0.25,
            Duration::Eighth => 0.125,
            Duration::Sixteenth => 0.0625,
            Duration::ThirtySecond => 0.03125,
            Duration::SixtyFourth => 0.015625,
            Duration::OneTwentyEighth => 0.0078125,
            Duration::DottedWhole => 1.5,
            Duration::DottedHalf => 0.75,
            Duration::DottedQuarter => 0.375,
            Duration::DottedEighth => 0.1875,
            Duration::DottedSixteenth => 0.09375,
            Duration::DottedThirtySecond => 0.046875,
            Duration::DottedSixtyFourth => 0.0234375,
            Duration::DottedOneTwentyEighth => 0.01171875,
            Duration::DoubleDottedWhole => 1.75,
            Duration::DoubleDottedHalf => 0.875,
            Duration::DoubleDottedQuarter => 0.4375,
            Duration::DoubleDottedEighth => 0.21875,
            Duration::DoubleDottedSixteenth => 0.109375,
            Duration::DoubleDottedThirtySecond => 0.0546875,
            Duration::DoubleDottedSixtyFourth => 0.02734375,
            Duration::DoubleDottedOneTwentyEighth => 0.013671875,
            Duration::TripleDottedWhole => 1.875,
            Duration::TripleDottedHalf => 0.9375,
            Duration::TripleDottedQuarter => 0.46875,
            Duration::TripleDottedEighth => 0.234375,
            Duration::TripleDottedSixteenth => 0.1171875,
            Duration::TripleDottedThirtySecond => 0.05859375,
            Duration::TripleDottedSixtyFourth => 0.029296875,
            Duration::TripleDottedOneTwentyEighth => 0.0146484375,
        }
    }
}

/// The core Note type - represents all musical constructs
#[derive(Clone, Debug)]
pub enum Note {
    /// A single pitched note
    Atom {
        midi: u8,
        duration: Duration,
        velocity: f32,
    },

    /// A rest (silence)
    Rest {
        duration: Duration,
    },

    /// Sequential composition - notes play one after another
    Serial(Vec<Note>),

    /// Parallel composition - notes play simultaneously (longest duration wins)
    Parallel(Vec<Note>),

    /// Parallel Min - notes play simultaneously (shortest duration wins)
    ParallelMin(Vec<Note>),

    /// Fork Sequence - plays in parallel with next note but doesn't advance timeline
    ForkSequence(Vec<Note>),

    /// Fork Parallel - parallel notes that don't advance the outer timeline
    ForkParallel(Vec<Note>),

    /// Set a parameter for subsequent notes
    Param {
        key: String,
        value: ParamValue,
    },

    /// Repeat the contained notes N times
    Repeat {
        times: usize,
        content: Box<Note>,
    },

    /// Duration tie - extends the previous note's duration
    DurationTie {
        duration: f32,
    },

    /// Previous pitch - uses the pitch from the previous note
    PreviousPitch {
        duration: Option<Duration>,
    },

    /// Implicit duration note - uses duration from previous note
    ImplicitDuration {
        midi: u8,
        velocity: f32,
    },

    /// Comment - doesn't affect timing, useful for annotations
    Comment(String),

    /// Repeat marker - repeats the previous note/block N times
    RepeatMarker(usize),

    /// Envelope - modulates a parameter over time
    Envelope(Envelope),

    /// Exact frequency note - plays using nearest MIDI note + pitch bend compensation
    /// Generates: PitchBend + NoteOn + NoteOff + PitchBend(center)
    Freq {
        frequency: f32,
        duration: Duration,
        velocity: f32,
    },

    /// Frequency sweep - glides from start to end frequency using nearest MIDI note + pitch bend
    /// Generates: NoteOn + PitchBend events + NoteOff
    Sweep {
        start_freq: f32,
        end_freq: f32,
        duration: Duration,
        velocity: f32,
    },
}

/// Parameter values
#[derive(Clone, Debug)]
pub enum ParamValue {
    Number(f32),
    Text(String),
    Note(Box<Note>),       // For macro invocation
    Envelope(Envelope),    // For envelope-based modulation
    Unset,                 // Removes parameter from context
}

// =============================================================================
// KEY SIGNATURE SUPPORT
// =============================================================================

/// Parse a key signature string like "Cmaj", "amin", "Bbmaj", "f#min"
pub fn parse_key_signature(key: &str) -> Option<(u8, bool)> {
    let key = key.to_lowercase();
    let mut chars = key.chars().peekable();

    // Parse root note
    let root = match chars.next()? {
        'c' => 0,
        'd' => 2,
        'e' => 4,
        'f' => 5,
        'g' => 7,
        'a' => 9,
        'b' => 11,
        _ => return None,
    };

    // Parse optional accidental
    let root = match chars.peek() {
        Some('#') | Some('s') => { chars.next(); (root + 1) % 12 }
        Some('b') | Some('f') => { chars.next(); (root + 11) % 12 }
        _ => root,
    };

    // Parse mode
    let remaining: String = chars.collect();
    let is_major = match remaining.as_str() {
        "maj" | "" => true,
        "min" | "m" => false,
        _ => return None,
    };

    Some((root, is_major))
}

/// Get the adjustment for a pitch class in a given key signature
pub fn get_key_adjustment(key: &str, pitch_class: u8) -> i8 {
    let Some((root, is_major)) = parse_key_signature(key) else {
        return 0;
    };

    // Scale degrees for major and minor
    let scale = if is_major {
        [0, 2, 4, 5, 7, 9, 11]  // Major scale
    } else {
        [0, 2, 3, 5, 7, 8, 10]  // Natural minor scale
    };

    // Check if pitch class is in the scale
    let relative_pitch = (pitch_class as i8 - root as i8).rem_euclid(12) as u8;

    if scale.contains(&relative_pitch) {
        0
    } else if scale.contains(&((relative_pitch + 1) % 12)) {
        -1  // Flat
    } else {
        1   // Sharp
    }
}

// =============================================================================
// TRANSPOSE SUPPORT
// =============================================================================

impl Note {
    /// Transpose a note by a number of semitones
    pub fn transpose(&self, semitones: i8) -> Note {
        match self {
            Note::Atom { midi, duration, velocity } => {
                let new_midi = (*midi as i16 + semitones as i16).clamp(0, 127) as u8;
                Note::Atom {
                    midi: new_midi,
                    duration: *duration,
                    velocity: *velocity,
                }
            }
            Note::ImplicitDuration { midi, velocity } => {
                let new_midi = (*midi as i16 + semitones as i16).clamp(0, 127) as u8;
                Note::ImplicitDuration {
                    midi: new_midi,
                    velocity: *velocity,
                }
            }
            Note::Serial(notes) => {
                Note::Serial(notes.iter().map(|n| n.transpose(semitones)).collect())
            }
            Note::Parallel(notes) => {
                Note::Parallel(notes.iter().map(|n| n.transpose(semitones)).collect())
            }
            Note::ParallelMin(notes) => {
                Note::ParallelMin(notes.iter().map(|n| n.transpose(semitones)).collect())
            }
            Note::ForkSequence(notes) => {
                Note::ForkSequence(notes.iter().map(|n| n.transpose(semitones)).collect())
            }
            Note::ForkParallel(notes) => {
                Note::ForkParallel(notes.iter().map(|n| n.transpose(semitones)).collect())
            }
            Note::Repeat { times, content } => {
                Note::Repeat {
                    times: *times,
                    content: Box::new(content.transpose(semitones)),
                }
            }
            // These don't need transposition
            Note::Rest { .. } | Note::Param { .. } | Note::DurationTie { .. } |
            Note::PreviousPitch { .. } | Note::Comment(_) | Note::RepeatMarker(_) |
            Note::Envelope(_) => {
                self.clone()
            }
            // Freq/Sweep use frequencies, transpose by ratio
            Note::Freq { frequency, duration, velocity } => {
                let ratio = 2.0_f32.powf(semitones as f32 / 12.0);
                Note::Freq {
                    frequency: frequency * ratio,
                    duration: *duration,
                    velocity: *velocity,
                }
            }
            Note::Sweep { start_freq, end_freq, duration, velocity } => {
                let ratio = 2.0_f32.powf(semitones as f32 / 12.0);
                Note::Sweep {
                    start_freq: start_freq * ratio,
                    end_freq: end_freq * ratio,
                    duration: *duration,
                    velocity: *velocity,
                }
            }
        }
    }

    /// Get total duration in whole notes
    pub fn total_duration(&self) -> f32 {
        match self {
            Note::Atom { duration, .. } => duration.value(),
            Note::Rest { duration } => duration.value(),
            Note::Serial(notes) => notes.iter().map(|n| n.total_duration()).sum(),
            Note::Parallel(notes) | Note::ParallelMin(notes) => {
                notes.iter().map(|n| n.total_duration()).fold(0.0, f32::max)
            }
            Note::ForkSequence(_) | Note::ForkParallel(_) => 0.0, // Don't advance timeline
            Note::Param { .. } => 0.0,
            Note::Repeat { times, content } => content.total_duration() * (*times as f32),
            Note::DurationTie { duration } => *duration,
            Note::PreviousPitch { duration } => duration.map(|d| d.value()).unwrap_or(0.25),
            Note::ImplicitDuration { .. } => 0.25, // Default to quarter
            Note::Comment(_) => 0.0,
            Note::RepeatMarker(_) => 0.0,
            Note::Envelope(_) => 0.0, // Envelopes don't advance timeline directly
            Note::Freq { duration, .. } => duration.value(),
            Note::Sweep { duration, .. } => duration.value(),
        }
    }
}

// =============================================================================
// GENERATED NOTE CONSTANTS
// =============================================================================

// Generate all note constants for octaves -1 to 9
note_macro::generate_all_notes!();

// Generate rest constants
note_macro::generate_rests!();

// Generate previous pitch constants
note_macro::generate_previous_pitch!();

// =============================================================================
// COMPOSITION FUNCTIONS
// =============================================================================

/// Sequential composition - notes play one after another
pub fn ser<I: IntoIterator<Item = Note>>(notes: I) -> Note {
    Note::Serial(notes.into_iter().collect())
}

/// Parallel composition - notes play simultaneously (longest duration wins)
pub fn par<I: IntoIterator<Item = Note>>(notes: I) -> Note {
    Note::Parallel(notes.into_iter().collect())
}

/// Parallel Min - notes play simultaneously (shortest duration wins)
pub fn parmin<I: IntoIterator<Item = Note>>(notes: I) -> Note {
    Note::ParallelMin(notes.into_iter().collect())
}

/// Fork Sequence - plays sequence without advancing outer timeline
pub fn forkseq<I: IntoIterator<Item = Note>>(notes: I) -> Note {
    Note::ForkSequence(notes.into_iter().collect())
}

/// Fork Parallel - plays parallel notes without advancing outer timeline
pub fn forkpar<I: IntoIterator<Item = Note>>(notes: I) -> Note {
    Note::ForkParallel(notes.into_iter().collect())
}

/// Serial composition macro: `ser![c4q, d4q, e4h]`
#[macro_export]
macro_rules! ser {
    [$($note:expr),* $(,)?] => {
        $crate::ser(vec![$($note),*])
    };
}

/// Parallel composition macro: `par![c4w, e4w, g4w]` for chords
#[macro_export]
macro_rules! par {
    [$($note:expr),* $(,)?] => {
        $crate::par(vec![$($note),*])
    };
}

/// Parallel Min macro: `parmin![c4e, e4q, g4h]` - advances by shortest
#[macro_export]
macro_rules! parmin {
    [$($note:expr),* $(,)?] => {
        $crate::parmin(vec![$($note),*])
    };
}

/// Fork Sequence macro: `forkseq![c4h, d4h]` - silent fork
#[macro_export]
macro_rules! forkseq {
    [$($note:expr),* $(,)?] => {
        $crate::forkseq(vec![$($note),*])
    };
}

/// Fork Parallel macro: `forkpar![c4q, c5q]` - silent parallel fork
#[macro_export]
macro_rules! forkpar {
    [$($note:expr),* $(,)?] => {
        $crate::forkpar(vec![$($note),*])
    };
}

/// Fork Sequence macro (alias for forkseq): `forkser![c4h, d4h]`
#[macro_export]
macro_rules! forkser {
    [$($note:expr),* $(,)?] => {
        $crate::forkseq(vec![$($note),*])
    };
}

// =============================================================================
// GENERATOR MACROS (for infinite sequences)
// =============================================================================

/// Re-export genawaiter's yield_ for convenience
pub use genawaiter::yield_;

/// Create a generator that yields Notes
///
/// # Example
/// ```ignore
/// use music::prelude::*;
///
/// let infinite_arpeggio = music_gen!({
///     loop {
///         yield_!(c4q);
///         yield_!(e4q);
///         yield_!(g4q);
///     }
/// });
/// ```
#[macro_export]
macro_rules! music_gen {
    ($body:block) => {{
        use $crate::genawaiter::sync::{Gen, Co};
        Gen::new(|co: Co<$crate::Note>| async move {
            // Bring yield_ into scope for the body
            macro_rules! yield_ {
                ($note:expr) => {
                    co.yield_($note).await
                };
            }
            $body
        }).into_iter()
    }};
}

// =============================================================================
// ENVELOPE MACROS
// =============================================================================

/// Creates an envelope with linear interpolation
///
/// # Format
/// Each point has the format: `<duration><value>` where:
/// - duration: note value (w, h, q, e, i, t, x, o and dotted variants)
///   or time value (e.g., 0.5s, 100ms, 50p for percent)
/// - value: the target value at this point (f32)
///
/// # Examples
/// ```ignore
/// // Attack-decay-sustain envelope using note durations
/// env!(q1.0, e0.7)  // Quarter note to 1.0, then eighth note to 0.7
///
/// // ADSR envelope using time values
/// env!(50ms1.0, 100ms0.7, 200ms0.7, 100ms0.0)
///
/// // Using percentages of total note duration
/// env!(10p1.0, 20p0.8, 70p0.8, 100p0.0)
/// ```
#[macro_export]
macro_rules! env {
    ($($point:expr),* $(,)?) => {{
        let mut points = Vec::new();
        $(
            points.push($crate::parse_envelope_point(stringify!($point)));
        )*
        $crate::Note::Envelope($crate::Envelope {
            target: String::new(),
            points,
            interpolation: $crate::InterpolationType::Linear,
        })
    }};
}

/// Creates an envelope with linear interpolation (alias for env!)
#[macro_export]
macro_rules! linen {
    ($($point:expr),* $(,)?) => {{
        $crate::env!($($point),*)
    }};
}

/// Creates an envelope with cosine interpolation for smoother transitions
#[macro_export]
macro_rules! cosen {
    ($($point:expr),* $(,)?) => {{
        let envelope = $crate::env!($($point),*);
        if let $crate::Note::Envelope(mut env) = envelope {
            env.interpolation = $crate::InterpolationType::Cosine;
            $crate::Note::Envelope(env)
        } else {
            envelope
        }
    }};
}

/// Creates an envelope with exponential interpolation
#[macro_export]
macro_rules! expen {
    ($($point:expr),* $(,)?) => {{
        let envelope = $crate::env!($($point),*);
        if let $crate::Note::Envelope(mut env) = envelope {
            env.interpolation = $crate::InterpolationType::Exponential;
            $crate::Note::Envelope(env)
        } else {
            envelope
        }
    }};
}

/// Creates an envelope with cubic interpolation for smoother, more natural transitions
#[macro_export]
macro_rules! cuben {
    ($($point:expr),* $(,)?) => {{
        let envelope = $crate::env!($($point),*);
        if let $crate::Note::Envelope(mut env) = envelope {
            env.interpolation = $crate::InterpolationType::Cubic;
            $crate::Note::Envelope(env)
        } else {
            envelope
        }
    }};
}

/// Set a parameter with an optional envelope value
///
/// # Examples
/// ```ignore
/// // Set a numeric parameter
/// param!(velocity = 0.8)
///
/// // Set a parameter with an envelope
/// param!(volume = env!(q1.0, h0.5))
/// ```
#[macro_export]
macro_rules! param {
    ($key:ident = $value:expr) => {{
        let value = $value;
        // Check if value is an envelope Note
        if let $crate::Note::Envelope(env) = &value {
            let mut env_with_target = env.clone();
            env_with_target.target = stringify!($key).to_string();
            $crate::Note::Param {
                key: stringify!($key).to_string(),
                value: $crate::ParamValue::Envelope(env_with_target),
            }
        } else {
            // For non-envelope values, try to convert to ParamValue
            $crate::Note::Param {
                key: stringify!($key).to_string(),
                value: $crate::param_value_from(value),
            }
        }
    }};
}

/// Helper function to convert various types to ParamValue
pub fn param_value_from<T: Into<ParamValueConvertible>>(value: T) -> ParamValue {
    value.into().0
}

/// Wrapper for ParamValue conversion
pub struct ParamValueConvertible(pub ParamValue);

impl From<f32> for ParamValueConvertible {
    fn from(v: f32) -> Self {
        ParamValueConvertible(ParamValue::Number(v))
    }
}

impl From<i32> for ParamValueConvertible {
    fn from(v: i32) -> Self {
        ParamValueConvertible(ParamValue::Number(v as f32))
    }
}

impl From<&str> for ParamValueConvertible {
    fn from(v: &str) -> Self {
        ParamValueConvertible(ParamValue::Text(v.to_string()))
    }
}

impl From<String> for ParamValueConvertible {
    fn from(v: String) -> Self {
        ParamValueConvertible(ParamValue::Text(v))
    }
}

impl From<Note> for ParamValueConvertible {
    fn from(v: Note) -> Self {
        if let Note::Envelope(env) = v {
            ParamValueConvertible(ParamValue::Envelope(env))
        } else {
            ParamValueConvertible(ParamValue::Note(Box::new(v)))
        }
    }
}

/// Parse a value string that may be relative (+X or -X) or absolute (X or =X)
/// Returns (value, is_relative)
fn parse_envelope_value(value_str: &str) -> Option<(f32, bool)> {
    let value_str = value_str.trim();
    if value_str.is_empty() {
        return None;
    }

    let first_char = value_str.chars().next()?;

    // +X or -X means relative
    if first_char == '+' || first_char == '-' {
        // Check if it's a relative value (+ or - followed by a number)
        // +0.5 = relative +0.5
        // -0.5 = relative -0.5
        if let Ok(value) = value_str.parse::<f32>() {
            return Some((value, true));
        }
    }

    // =X means explicit absolute (useful for =-0.5)
    if first_char == '=' {
        if let Ok(value) = value_str[1..].parse::<f32>() {
            return Some((value, false));
        }
    }

    // Plain number is absolute
    if let Ok(value) = value_str.parse::<f32>() {
        return Some((value, false));
    }

    None
}

/// Parse an envelope point from a string representation
///
/// Supports formats:
/// - Note duration + value: "q1.0", "hd0.5", "e0.8"
/// - Time in seconds: "0.5s1.0", "1.0s0.0"
/// - Time in milliseconds: "100ms1.0", "50ms0.5"
/// - Percentage: "50p1.0", "100p0.0"
///
/// Values can be absolute or relative:
/// - Absolute: "q1.0", "q0.5", "q=-0.5" (= prefix for explicit absolute)
/// - Relative: "q+0.5" (add 0.5), "q-0.3" (subtract 0.3)
///
/// Examples:
/// - `env!(q1.0, q+0.2, q-0.1)` → 1.0, then 1.2, then 1.1
/// - `env!(0s0.0, 100ms+0.5, 100ms-0.5)` → vibrato-like oscillation
pub fn parse_envelope_point(point_str: &str) -> EnvelopePoint {
    let point_str = point_str.trim();

    // Metric duration codes
    let metric_codes = ["wddd", "hddd", "qddd", "eddd", "iddd", "tddd", "xddd", "oddd",
                        "wdd", "hdd", "qdd", "edd", "idd", "tdd", "xdd", "odd",
                        "wd", "hd", "qd", "ed", "id", "td", "xd", "od",
                        "w", "h", "q", "e", "i", "t", "x", "o"];

    // Try to match metric duration codes first
    for &code in &metric_codes {
        if point_str.starts_with(code) {
            let value_str = &point_str[code.len()..];
            if let Some((value, relative)) = parse_envelope_value(value_str) {
                let duration = match code {
                    "w" => Duration::Whole,
                    "h" => Duration::Half,
                    "q" => Duration::Quarter,
                    "e" => Duration::Eighth,
                    "i" => Duration::Sixteenth,
                    "t" => Duration::ThirtySecond,
                    "x" => Duration::SixtyFourth,
                    "o" => Duration::OneTwentyEighth,
                    "wd" => Duration::DottedWhole,
                    "hd" => Duration::DottedHalf,
                    "qd" => Duration::DottedQuarter,
                    "ed" => Duration::DottedEighth,
                    "id" => Duration::DottedSixteenth,
                    "td" => Duration::DottedThirtySecond,
                    "xd" => Duration::DottedSixtyFourth,
                    "od" => Duration::DottedOneTwentyEighth,
                    "wdd" => Duration::DoubleDottedWhole,
                    "hdd" => Duration::DoubleDottedHalf,
                    "qdd" => Duration::DoubleDottedQuarter,
                    "edd" => Duration::DoubleDottedEighth,
                    "idd" => Duration::DoubleDottedSixteenth,
                    "tdd" => Duration::DoubleDottedThirtySecond,
                    "xdd" => Duration::DoubleDottedSixtyFourth,
                    "odd" => Duration::DoubleDottedOneTwentyEighth,
                    "wddd" => Duration::TripleDottedWhole,
                    "hddd" => Duration::TripleDottedHalf,
                    "qddd" => Duration::TripleDottedQuarter,
                    "eddd" => Duration::TripleDottedEighth,
                    "iddd" => Duration::TripleDottedSixteenth,
                    "tddd" => Duration::TripleDottedThirtySecond,
                    "xddd" => Duration::TripleDottedSixtyFourth,
                    "oddd" => Duration::TripleDottedOneTwentyEighth,
                    _ => Duration::Quarter,
                };
                if relative {
                    return EnvelopePoint::new_note_relative(value, duration);
                } else {
                    return EnvelopePoint::new_note(value, duration);
                }
            }
        }
    }

    // Try to match time units: ms, s, p (must check ms before s)
    if let Some(ms_idx) = point_str.find("ms") {
        // Format: <time>ms<value>
        let time_str = &point_str[..ms_idx];
        let value_str = &point_str[ms_idx + 2..];
        if let Ok(time) = time_str.parse::<f32>() {
            if let Some((value, relative)) = parse_envelope_value(value_str) {
                if relative {
                    return EnvelopePoint::new_ms_relative(value, time);
                } else {
                    return EnvelopePoint::new_ms(value, time);
                }
            }
        }
    }

    if let Some(s_idx) = point_str.find('s') {
        // Format: <time>s<value>
        let time_str = &point_str[..s_idx];
        let value_str = &point_str[s_idx + 1..];
        if let Ok(time) = time_str.parse::<f32>() {
            if let Some((value, relative)) = parse_envelope_value(value_str) {
                if relative {
                    return EnvelopePoint::new_seconds_relative(value, time);
                } else {
                    return EnvelopePoint::new_seconds(value, time);
                }
            }
        }
    }

    if let Some(p_idx) = point_str.find('p') {
        // Format: <percent>p<value>
        let percent_str = &point_str[..p_idx];
        let value_str = &point_str[p_idx + 1..];
        if let Ok(percent) = percent_str.parse::<f32>() {
            if let Some((value, relative)) = parse_envelope_value(value_str) {
                if relative {
                    return EnvelopePoint::new_percent_relative(value, percent);
                } else {
                    return EnvelopePoint::new_percent(value, percent);
                }
            }
        }
    }

    // Fallback: try to parse as just a value with quarter note duration
    if let Some((value, relative)) = parse_envelope_value(point_str) {
        if relative {
            return EnvelopePoint::new_note_relative(value, Duration::Quarter);
        } else {
            return EnvelopePoint::new_note(value, Duration::Quarter);
        }
    }

    panic!("Invalid envelope point format: {}", point_str);
}

/// Repeat marker macro
#[macro_export]
macro_rules! repeat {
    ($count:expr) => {
        $crate::Note::RepeatMarker($count)
    };
}

/// Note with parameters macro
///
/// Wraps a note constant with note-specific parameters that apply only to that note.
/// The parameters are scoped to just this note using a Serial wrapper.
///
/// # Examples
/// ```ignore
/// // Note with volume and duty cycle
/// note!(c4q, volume = 0.8, duty = 0.25)
///
/// // Note with attack envelope
/// note!(c4h, attack_envelope = env!(q1.0, h0.5))
///
/// // Multiple parameters
/// note!(g4q, velocity = 0.9, gate = 0.8, patch = "sine")
/// ```
#[macro_export]
macro_rules! note {
    // Single note without parameters
    ($note:expr) => {
        $note
    };
    // Note with parameters - wraps in a Serial to scope the params
    ($note:expr, $($key:ident = $value:expr),+ $(,)?) => {{
        let mut notes: Vec<$crate::Note> = Vec::new();
        $(
            notes.push($crate::Note::Param {
                key: stringify!($key).to_string(),
                value: $crate::param_value_from($value),
            });
        )+
        notes.push($note);
        $crate::Note::Serial(notes)
    }};
}

/// Repeat a pattern N times
pub fn rep(times: usize, content: Note) -> Note {
    Note::Repeat {
        times,
        content: Box::new(content),
    }
}

/// Play an exact frequency using nearest MIDI note + pitch bend
/// Useful for frequencies that don't map exactly to MIDI notes
pub fn freq(frequency: f32, duration: Duration) -> Note {
    Note::Freq {
        frequency,
        duration,
        velocity: 0.8,
    }
}

/// Play an exact frequency with custom velocity
pub fn freq_vel(frequency: f32, duration: Duration, velocity: f32) -> Note {
    Note::Freq {
        frequency,
        duration,
        velocity,
    }
}

/// Sweep from one frequency to another over a duration
/// Generates pitch bend events for smooth glide
pub fn sweep(start_freq: f32, end_freq: f32, duration: Duration) -> Note {
    Note::Sweep {
        start_freq,
        end_freq,
        duration,
        velocity: 0.8,
    }
}

/// Sweep with custom velocity
pub fn sweep_vel(start_freq: f32, end_freq: f32, duration: Duration, velocity: f32) -> Note {
    Note::Sweep {
        start_freq,
        end_freq,
        duration,
        velocity,
    }
}

/// Repeat marker - repeats the previous note/block N times
pub fn repeat(times: usize) -> Note {
    Note::RepeatMarker(times)
}

/// Comment - annotation that doesn't affect timing
pub fn comment<S: Into<String>>(text: S) -> Note {
    Note::Comment(text.into())
}

/// Comment macro with format string support
#[macro_export]
macro_rules! comment {
    ($($arg:tt)*) => {
        $crate::comment(format!($($arg)*))
    };
}

/// Set tempo in BPM
pub fn tempo(bpm: f32) -> Note {
    Note::Param {
        key: "tempo".to_string(),
        value: ParamValue::Number(bpm),
    }
}

// =============================================================================
// GENERAL MIDI PROGRAM NUMBERS
// =============================================================================

/// General MIDI program numbers (0-127)
/// Compatible with SC-55, SC-88, and other GM-compliant sound modules
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum GM {
    // Piano (0-7)
    AcousticGrandPiano = 0,
    BrightAcousticPiano = 1,
    ElectricGrandPiano = 2,
    HonkyTonkPiano = 3,
    ElectricPiano1 = 4,
    ElectricPiano2 = 5,
    Harpsichord = 6,
    Clavinet = 7,

    // Chromatic Percussion (8-15)
    Celesta = 8,
    Glockenspiel = 9,
    MusicBox = 10,
    Vibraphone = 11,
    Marimba = 12,
    Xylophone = 13,
    TubularBells = 14,
    Dulcimer = 15,

    // Organ (16-23)
    DrawbarOrgan = 16,
    PercussiveOrgan = 17,
    RockOrgan = 18,
    ChurchOrgan = 19,
    ReedOrgan = 20,
    Accordion = 21,
    Harmonica = 22,
    TangoAccordion = 23,

    // Guitar (24-31)
    AcousticGuitarNylon = 24,
    AcousticGuitarSteel = 25,
    ElectricGuitarJazz = 26,
    ElectricGuitarClean = 27,
    ElectricGuitarMuted = 28,
    OverdrivenGuitar = 29,
    DistortionGuitar = 30,
    GuitarHarmonics = 31,

    // Bass (32-39)
    AcousticBass = 32,
    ElectricBassFinger = 33,
    ElectricBassPick = 34,
    FretlessBass = 35,
    SlapBass1 = 36,
    SlapBass2 = 37,
    SynthBass1 = 38,
    SynthBass2 = 39,

    // Strings (40-47)
    Violin = 40,
    Viola = 41,
    Cello = 42,
    Contrabass = 43,
    TremoloStrings = 44,
    PizzicatoStrings = 45,
    OrchestralHarp = 46,
    Timpani = 47,

    // Ensemble (48-55)
    StringEnsemble1 = 48,
    StringEnsemble2 = 49,
    SynthStrings1 = 50,
    SynthStrings2 = 51,
    ChoirAahs = 52,
    VoiceOohs = 53,
    SynthVoice = 54,
    OrchestraHit = 55,

    // Brass (56-63)
    Trumpet = 56,
    Trombone = 57,
    Tuba = 58,
    MutedTrumpet = 59,
    FrenchHorn = 60,
    BrassSection = 61,
    SynthBrass1 = 62,
    SynthBrass2 = 63,

    // Reed (64-71)
    SopranoSax = 64,
    AltoSax = 65,
    TenorSax = 66,
    BaritoneSax = 67,
    Oboe = 68,
    EnglishHorn = 69,
    Bassoon = 70,
    Clarinet = 71,

    // Pipe (72-79)
    Piccolo = 72,
    Flute = 73,
    Recorder = 74,
    PanFlute = 75,
    BlownBottle = 76,
    Shakuhachi = 77,
    Whistle = 78,
    Ocarina = 79,

    // Synth Lead (80-87)
    SquareLead = 80,
    SawtoothLead = 81,
    CalliopeLead = 82,
    ChiffLead = 83,
    CharangLead = 84,
    VoiceLead = 85,
    FifthsLead = 86,
    BassAndLead = 87,

    // Synth Pad (88-95)
    NewAgePad = 88,
    WarmPad = 89,
    PolysynthPad = 90,
    ChoirPad = 91,
    BowedPad = 92,
    MetallicPad = 93,
    HaloPad = 94,
    SweepPad = 95,

    // Synth Effects (96-103)
    Rain = 96,
    Soundtrack = 97,
    Crystal = 98,
    Atmosphere = 99,
    Brightness = 100,
    Goblins = 101,
    Echoes = 102,
    SciFi = 103,

    // Ethnic (104-111)
    Sitar = 104,
    Banjo = 105,
    Shamisen = 106,
    Koto = 107,
    Kalimba = 108,
    Bagpipe = 109,
    Fiddle = 110,
    Shanai = 111,

    // Percussive (112-119)
    TinkleBell = 112,
    Agogo = 113,
    SteelDrums = 114,
    Woodblock = 115,
    TaikoDrum = 116,
    MelodicTom = 117,
    SynthDrum = 118,
    ReverseCymbal = 119,

    // Sound Effects (120-127)
    GuitarFretNoise = 120,
    BreathNoise = 121,
    Seashore = 122,
    BirdTweet = 123,
    TelephoneRing = 124,
    Helicopter = 125,
    Applause = 126,
    Gunshot = 127,
}

impl From<GM> for u8 {
    fn from(gm: GM) -> u8 {
        gm as u8
    }
}

/// Set MIDI channel (0-15)
pub fn channel(ch: u8) -> Note {
    Note::Param {
        key: "channel".to_string(),
        value: ParamValue::Number(ch as f32),
    }
}

/// Set MIDI program/instrument (0-127)
/// Accepts either a u8 or a GM enum variant
pub fn program(prog: impl Into<u8>) -> Note {
    Note::Param {
        key: "program".to_string(),
        value: ParamValue::Number(prog.into() as f32),
    }
}

/// Set velocity (0.0 - 1.0)
pub fn velocity(vel: f32) -> Note {
    Note::Param {
        key: "velocity".to_string(),
        value: ParamValue::Number(vel),
    }
}

/// Set gate (0.0 - 1.0) - controls note sounding duration as fraction of written duration
pub fn gate(g: f32) -> Note {
    Note::Param {
        key: "gate".to_string(),
        value: ParamValue::Number(g.clamp(0.0, 1.0)),
    }
}

/// Set transpose offset in semitones
pub fn transpose(semitones: i8) -> Note {
    Note::Param {
        key: "transpose".to_string(),
        value: ParamValue::Number(semitones as f32),
    }
}

/// Set key signature (e.g., "Cmaj", "amin", "Bbmaj")
pub fn key<S: Into<String>>(key_sig: S) -> Note {
    Note::Param {
        key: "key".to_string(),
        value: ParamValue::Text(key_sig.into()),
    }
}

/// Set time note (beat unit, default 4 = quarter note)
pub fn time_note(note: u8) -> Note {
    Note::Param {
        key: "time_note".to_string(),
        value: ParamValue::Number(note as f32),
    }
}

/// Set a macro to play with notes
pub fn set_macro(macro_note: Note) -> Note {
    Note::Param {
        key: "macro".to_string(),
        value: ParamValue::Note(Box::new(macro_note)),
    }
}

/// Clear the macro
pub fn clear_macro() -> Note {
    Note::Param {
        key: "macro".to_string(),
        value: ParamValue::Unset,
    }
}

/// Set patch/waveform type (sine, square, triangle, sawtooth, noise)
pub fn patch<S: Into<String>>(patch_name: S) -> Note {
    Note::Param {
        key: "patch".to_string(),
        value: ParamValue::Text(patch_name.into()),
    }
}

/// Set duty cycle for square waves (0.0 - 1.0)
pub fn duty(duty_cycle: f32) -> Note {
    Note::Param {
        key: "duty".to_string(),
        value: ParamValue::Number(duty_cycle.clamp(0.0, 1.0)),
    }
}

/// Set noise type (white, periodic, brown, pink)
pub fn noise_type<S: Into<String>>(noise: S) -> Note {
    Note::Param {
        key: "noise_type".to_string(),
        value: ParamValue::Text(noise.into()),
    }
}

/// Set instrument name
pub fn instrument<S: Into<String>>(name: S) -> Note {
    Note::Param {
        key: "instrument".to_string(),
        value: ParamValue::Text(name.into()),
    }
}

/// Set volume (0.0 - 1.0)
pub fn volume(vol: f32) -> Note {
    Note::Param {
        key: "volume".to_string(),
        value: ParamValue::Number(vol.clamp(0.0, 1.0)),
    }
}

/// Set attack envelope for note onset
pub fn attack_envelope(env: Envelope) -> Note {
    Note::Param {
        key: "attack_envelope".to_string(),
        value: ParamValue::Envelope(env),
    }
}

/// Set sustain envelope for note sustain phase
pub fn sustain_envelope(env: Envelope) -> Note {
    Note::Param {
        key: "sustain_envelope".to_string(),
        value: ParamValue::Envelope(env),
    }
}

/// Set release envelope for note release
pub fn release_envelope(env: Envelope) -> Note {
    Note::Param {
        key: "release_envelope".to_string(),
        value: ParamValue::Envelope(env),
    }
}

/// Set release duration in seconds
pub fn release_duration(seconds: f32) -> Note {
    Note::Param {
        key: "release_duration".to_string(),
        value: ParamValue::Number(seconds),
    }
}

/// Set envelope preset (e.g., "percussion")
pub fn envelope_preset<S: Into<String>>(preset: S) -> Note {
    Note::Param {
        key: "envelope_preset".to_string(),
        value: ParamValue::Text(preset.into()),
    }
}

// =============================================================================
// FREQUENCY/PITCH BEND HELPERS
// =============================================================================

/// Convert MIDI note to frequency
pub fn midi_to_freq(midi_note: u8) -> f32 {
    440.0 * 2.0_f32.powf((midi_note as f32 - 69.0) / 12.0)
}

/// Convert frequency to nearest MIDI note and pitch bend value
/// Returns (midi_note, pitch_bend_value) where pitch_bend is 14-bit (8192 = center)
pub fn freq_to_midi_with_bend(freq: f32, bend_range_semitones: f32) -> (u8, u16) {
    // Calculate exact MIDI note (can be fractional)
    let exact_midi = 69.0 + 12.0 * (freq / 440.0).log2();
    let nearest_midi = exact_midi.round().clamp(0.0, 127.0) as u8;

    // Calculate semitone difference from nearest note
    let semitone_diff = exact_midi - nearest_midi as f32;

    // Convert to pitch bend
    let bend_value = semitones_to_pitch_bend(semitone_diff, bend_range_semitones);

    (nearest_midi, bend_value)
}

/// Convert semitone offset to 14-bit pitch bend value
/// bend_range_semitones is the synth's pitch bend range (e.g., 48 semitones)
pub fn semitones_to_pitch_bend(semitones: f32, bend_range_semitones: f32) -> u16 {
    let normalized = semitones / bend_range_semitones;
    ((normalized * 8192.0) as i16 + 8192).clamp(0, 16383) as u16
}

// =============================================================================
// MIDI EVENT TYPES
// =============================================================================

/// A MIDI event with timing
#[derive(Clone, Debug)]
pub struct MidiEvent {
    pub time: f32,
    pub channel: u8,
    pub event_type: MidiEventType,
}

#[derive(Clone, Debug)]
pub enum MidiEventType {
    NoteOn { note: u8, velocity: u8 },
    NoteOff { note: u8 },
    ProgramChange { program: u8 },
    ControlChange { controller: u8, value: u8 },
    PitchBend { value: u16 }, // 14-bit value, 8192 = center
    Comment(String),
}

// =============================================================================
// STREAMING EVENT GENERATION (using genawaiter)
// =============================================================================

use genawaiter::sync::{Co, Gen};
use async_recursion::async_recursion;

/// Iterator over Notes - allows lazy note generation
pub type NoteIterator = Box<dyn Iterator<Item = Note> + 'static>;

/// Iterator over MidiEvents - allows lazy event generation
pub type MidiEventIterator = Box<dyn Iterator<Item = MidiEvent> + 'static>;

/// Parameters that affect note generation
#[derive(Clone)]
pub struct Params {
    pub tempo: f32,
    pub time_note: f32,
    pub channel: u8,
    pub program: u8,
    pub velocity: f32,
    pub volume: f32,
    pub gate: f32,
    pub transpose: i8,
    pub key: Option<String>,
    pub last_midi: Option<u8>,
    pub last_duration: Duration,
    pub active_macro: Option<Box<Note>>,
    // Synthesis parameters
    pub patch: String,
    pub duty: f32,
    pub noise_type: String,
    pub instrument: String,
    // ADSR envelope parameters
    pub attack_envelope: Option<Envelope>,
    pub sustain_envelope: Option<Envelope>,
    pub release_envelope: Option<Envelope>,
    pub release_duration: f32,
    pub envelope_preset: Option<String>,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            tempo: 120.0,
            time_note: 4.0,
            channel: 0,
            program: 0,
            velocity: 0.8,
            volume: 1.0,
            gate: 0.9,
            transpose: 0,
            key: None,
            last_midi: None,
            last_duration: Duration::Quarter,
            active_macro: None,
            patch: "square".to_string(),
            duty: 0.5,
            noise_type: "white".to_string(),
            instrument: "basic_synth".to_string(),
            attack_envelope: None,
            sustain_envelope: None,
            release_envelope: None,
            release_duration: 0.1,
            envelope_preset: None,
        }
    }
}

impl Params {
    /// Convert metric duration (fraction of whole note) to seconds
    pub fn metric_to_seconds(&self, metric_duration: f32) -> f32 {
        // whole_note_duration = (60 / tempo) * time_note
        let whole_note_secs = (60.0 / self.tempo) * self.time_note;
        metric_duration * whole_note_secs
    }
}

impl Note {
    /// Create a lazy streaming iterator over MIDI events
    /// Events are generated on-demand as the iterator is consumed
    pub fn event_stream(&self, tempo: f32) -> impl Iterator<Item = MidiEvent> + 'static {
        let note = self.clone();
        let mut params = Params::default();
        params.tempo = tempo;

        Gen::new(move |co: Co<MidiEvent>| async move {
            generate_events_recursive(&note, 0.0, &mut params.clone(), &co).await;
        }).into_iter()
    }

    /// Create a lazy streaming iterator with custom initial parameters
    pub fn event_stream_with_params(&self, params: Params) -> impl Iterator<Item = MidiEvent> + 'static {
        let note = self.clone();

        Gen::new(move |co: Co<MidiEvent>| async move {
            generate_events_recursive(&note, 0.0, &mut params.clone(), &co).await;
        }).into_iter()
    }

    /// Create an infinite looping iterator over MIDI events
    /// The composition repeats forever with proper time offsets
    pub fn event_stream_looping(&self, tempo: f32) -> impl Iterator<Item = MidiEvent> + 'static {
        let note = self.clone();
        let mut params = Params::default();
        params.tempo = tempo;

        Gen::new(move |co: Co<MidiEvent>| async move {
            let mut loop_start = 0.0f32;

            loop {
                let mut loop_params = params.clone();
                let end_time = generate_events_recursive_with_offset(
                    &note, 0.0, loop_start, &mut loop_params, &co
                ).await;

                // Next loop starts after this one ends
                loop_start = end_time;
            }
        }).into_iter()
    }
}

/// Convert a stream of Notes to a stream of MidiEvents
/// Useful for processing notes from an external source (file, network, etc.)
pub fn note_stream_to_event_stream(note_stream: NoteIterator, tempo: f32) -> MidiEventIterator {
    let mut params = Params::default();
    params.tempo = tempo;

    let generator = Gen::new(move |co: Co<MidiEvent>| async move {
        let mut current_time = 0.0f32;

        for note in note_stream {
            // Generate events for this note
            let end_time = generate_events_recursive(&note, current_time, &mut params, &co).await;
            current_time = end_time;
        }
    });

    Box::new(generator.into_iter())
}

/// Recursively generate MIDI events, yielding each one lazily
#[async_recursion(?Send)]
async fn generate_events_recursive(
    note: &Note,
    start_time: f32,
    params: &mut Params,
    co: &Co<MidiEvent>,
) -> f32 {
    generate_events_recursive_with_offset(note, start_time, 0.0, params, co).await
}

/// Generate events with a time offset (for looping)
#[async_recursion(?Send)]
async fn generate_events_recursive_with_offset(
    note: &Note,
    start_time: f32,
    time_offset: f32,
    params: &mut Params,
    co: &Co<MidiEvent>,
) -> f32 {
    match note {
        Note::Atom { midi, duration, velocity } => {
            let transposed_midi = (*midi as i16 + params.transpose as i16).clamp(0, 127) as u8;
            let dur_secs = params.metric_to_seconds(duration.value());
            let vel_byte = (velocity * params.velocity * 127.0).clamp(0.0, 127.0) as u8;
            let gate_dur = dur_secs * params.gate;

            // Yield NoteOn
            co.yield_(MidiEvent {
                time: start_time + time_offset,
                channel: params.channel,
                event_type: MidiEventType::NoteOn { note: transposed_midi, velocity: vel_byte },
            }).await;

            // Yield NoteOff
            co.yield_(MidiEvent {
                time: start_time + time_offset + gate_dur,
                channel: params.channel,
                event_type: MidiEventType::NoteOff { note: transposed_midi },
            }).await;

            // Handle macro invocation
            if let Some(macro_note) = &params.active_macro {
                let transpose_offset = transposed_midi as i8 - 60;
                let transposed_macro = macro_note.transpose(transpose_offset);
                let mut macro_params = params.clone();
                macro_params.active_macro = None;
                generate_events_recursive_with_offset(&transposed_macro, start_time, time_offset, &mut macro_params, co).await;
            }

            params.last_midi = Some(*midi);
            params.last_duration = *duration;

            start_time + dur_secs
        }

        Note::Rest { duration } => {
            params.last_duration = *duration;
            start_time + params.metric_to_seconds(duration.value())
        }

        Note::Serial(notes) => {
            let mut current_time = start_time;
            let mut idx = 0;

            while idx < notes.len() {
                let note = &notes[idx];

                // Handle repeat markers
                if let Note::RepeatMarker(times) = note {
                    if idx > 0 {
                        let prev_note = &notes[idx - 1];
                        for _ in 0..*times {
                            current_time = generate_events_recursive_with_offset(
                                prev_note, current_time, time_offset, params, co
                            ).await;
                        }
                    }
                    idx += 1;
                    continue;
                }

                // Handle duration ties - extend previous note
                if let Note::DurationTie { duration: tie_dur } = note {
                    current_time += params.metric_to_seconds(*tie_dur);
                    idx += 1;
                    continue;
                }

                current_time = generate_events_recursive_with_offset(
                    note, current_time, time_offset, params, co
                ).await;
                idx += 1;
            }
            current_time
        }

        Note::Parallel(notes) => {
            let mut max_end_time = start_time;
            for n in notes {
                let mut branch_params = params.clone();
                let end_time = generate_events_recursive_with_offset(
                    n, start_time, time_offset, &mut branch_params, co
                ).await;
                max_end_time = max_end_time.max(end_time);
            }
            max_end_time
        }

        Note::ParallelMin(notes) => {
            let mut min_end_time = f32::MAX;
            for n in notes {
                let mut branch_params = params.clone();
                let end_time = generate_events_recursive_with_offset(
                    n, start_time, time_offset, &mut branch_params, co
                ).await;
                if end_time > start_time {
                    min_end_time = min_end_time.min(end_time);
                }
            }
            if min_end_time == f32::MAX { start_time } else { min_end_time }
        }

        Note::ForkSequence(notes) => {
            let mut fork_params = params.clone();
            let mut current_time = start_time;
            for n in notes {
                current_time = generate_events_recursive_with_offset(
                    n, current_time, time_offset, &mut fork_params, co
                ).await;
            }
            start_time // Fork doesn't advance outer timeline
        }

        Note::ForkParallel(notes) => {
            for n in notes {
                let mut fork_params = params.clone();
                generate_events_recursive_with_offset(
                    n, start_time, time_offset, &mut fork_params, co
                ).await;
            }
            start_time // Fork doesn't advance outer timeline
        }

        Note::Param { key, value } => {
            match (key.as_str(), value) {
                // Core timing/control parameters
                ("tempo", ParamValue::Number(v)) => params.tempo = *v,
                ("time_note", ParamValue::Number(v)) => params.time_note = *v,
                ("channel", ParamValue::Number(v)) => params.channel = *v as u8,
                ("velocity", ParamValue::Number(v)) => params.velocity = *v,
                ("volume", ParamValue::Number(v)) => params.volume = *v,
                ("gate", ParamValue::Number(v)) => params.gate = *v,
                ("transpose", ParamValue::Number(v)) => params.transpose = *v as i8,
                ("key", ParamValue::Text(k)) => params.key = Some(k.clone()),
                ("key", ParamValue::Unset) => params.key = None,

                // Program change
                ("program", ParamValue::Number(v)) => {
                    params.program = *v as u8;
                    co.yield_(MidiEvent {
                        time: start_time + time_offset,
                        channel: params.channel,
                        event_type: MidiEventType::ProgramChange { program: params.program },
                    }).await;
                }

                // Synthesis parameters
                ("patch", ParamValue::Text(patch_name)) => params.patch = patch_name.clone(),
                ("duty", ParamValue::Number(v)) => params.duty = *v,
                ("noise_type", ParamValue::Text(noise)) => params.noise_type = noise.clone(),
                ("instrument", ParamValue::Text(instr)) => params.instrument = instr.clone(),

                // ADSR envelope parameters
                ("attack_envelope", ParamValue::Envelope(attack_env)) => {
                    params.attack_envelope = Some(attack_env.clone());
                }
                ("attack_envelope", ParamValue::Unset) => {
                    params.attack_envelope = None;
                }
                ("sustain_envelope", ParamValue::Envelope(sustain_env)) => {
                    params.sustain_envelope = Some(sustain_env.clone());
                }
                ("sustain_envelope", ParamValue::Unset) => {
                    params.sustain_envelope = None;
                }
                ("release_envelope", ParamValue::Envelope(release_env)) => {
                    params.release_envelope = Some(release_env.clone());
                }
                ("release_envelope", ParamValue::Unset) => {
                    params.release_envelope = None;
                }
                ("release_duration", ParamValue::Number(v)) => params.release_duration = *v,
                ("envelope_preset", ParamValue::Text(preset_name)) => {
                    params.envelope_preset = Some(preset_name.clone());
                }
                ("envelope_preset", ParamValue::Unset) => {
                    params.envelope_preset = None;
                }

                // Note macro
                ("macro", ParamValue::Note(n)) => {
                    params.active_macro = Some(n.clone());
                }
                ("macro", ParamValue::Unset) => {
                    params.active_macro = None;
                }
                _ => {}
            }
            start_time
        }

        Note::Repeat { times, content } => {
            let mut current_time = start_time;
            for _ in 0..*times {
                current_time = generate_events_recursive_with_offset(
                    content, current_time, time_offset, params, co
                ).await;
            }
            current_time
        }

        Note::DurationTie { duration } => {
            start_time + params.metric_to_seconds(*duration)
        }

        Note::PreviousPitch { duration } => {
            if let Some(midi) = params.last_midi {
                let dur = duration.unwrap_or(params.last_duration);
                let transposed_midi = (midi as i16 + params.transpose as i16).clamp(0, 127) as u8;
                let dur_secs = params.metric_to_seconds(dur.value());
                let vel_byte = (0.8 * params.velocity * 127.0).clamp(0.0, 127.0) as u8;
                let gate_dur = dur_secs * params.gate;

                co.yield_(MidiEvent {
                    time: start_time + time_offset,
                    channel: params.channel,
                    event_type: MidiEventType::NoteOn { note: transposed_midi, velocity: vel_byte },
                }).await;

                co.yield_(MidiEvent {
                    time: start_time + time_offset + gate_dur,
                    channel: params.channel,
                    event_type: MidiEventType::NoteOff { note: transposed_midi },
                }).await;

                params.last_duration = dur;
                start_time + dur_secs
            } else {
                start_time
            }
        }

        Note::ImplicitDuration { midi, velocity } => {
            let dur = params.last_duration;
            let transposed_midi = (*midi as i16 + params.transpose as i16).clamp(0, 127) as u8;
            let dur_secs = params.metric_to_seconds(dur.value());
            let vel_byte = (velocity * params.velocity * 127.0).clamp(0.0, 127.0) as u8;
            let gate_dur = dur_secs * params.gate;

            co.yield_(MidiEvent {
                time: start_time + time_offset,
                channel: params.channel,
                event_type: MidiEventType::NoteOn { note: transposed_midi, velocity: vel_byte },
            }).await;

            co.yield_(MidiEvent {
                time: start_time + time_offset + gate_dur,
                channel: params.channel,
                event_type: MidiEventType::NoteOff { note: transposed_midi },
            }).await;

            params.last_midi = Some(*midi);
            start_time + dur_secs
        }

        Note::Comment(text) => {
            co.yield_(MidiEvent {
                time: start_time + time_offset,
                channel: 0,
                event_type: MidiEventType::Comment(text.clone()),
            }).await;
            start_time
        }

        Note::RepeatMarker(_) => {
            start_time // Handled in Serial processing
        }

        Note::Envelope(envelope) => {
            // Envelopes are typically applied to parameters via Note::Param
            // When encountered standalone, we could generate CC messages for modulation
            // For now, we store the envelope in params for future note processing
            // The envelope will be evaluated per-note based on its target parameter

            // Generate control change events for the envelope if it targets volume
            if envelope.target == "volume" || envelope.target == "velocity" {
                let total_dur = envelope.total_duration(params.tempo, params.time_note);
                let steps = (total_dur * 20.0) as usize; // ~20 CC messages per second
                let step_duration = if steps > 0 { total_dur / steps as f32 } else { 0.0 };

                for step in 0..=steps {
                    let step_time = step as f32 * step_duration;
                    let value = envelope.evaluate(step_time, params.tempo, params.time_note, Some(total_dur));
                    let cc_value = (value * 127.0).clamp(0.0, 127.0) as u8;

                    co.yield_(MidiEvent {
                        time: start_time + time_offset + step_time,
                        channel: params.channel,
                        event_type: MidiEventType::ControlChange {
                            controller: 7, // Volume CC
                            value: cc_value,
                        },
                    }).await;
                }

                start_time + total_dur
            } else if envelope.target == "pitch" {
                // Generate pitch bend events
                let total_dur = envelope.total_duration(params.tempo, params.time_note);
                let steps = (total_dur * 20.0) as usize;
                let step_duration = if steps > 0 { total_dur / steps as f32 } else { 0.0 };

                for step in 0..=steps {
                    let step_time = step as f32 * step_duration;
                    let value = envelope.evaluate(step_time, params.tempo, params.time_note, Some(total_dur));
                    // Convert to 14-bit pitch bend (8192 = center)
                    let bend_value = ((value * 8192.0) as i16 + 8192).clamp(0, 16383) as u16;

                    co.yield_(MidiEvent {
                        time: start_time + time_offset + step_time,
                        channel: params.channel,
                        event_type: MidiEventType::PitchBend { value: bend_value },
                    }).await;
                }

                start_time + total_dur
            } else {
                // For other envelope targets, just advance time by envelope duration
                start_time + envelope.total_duration(params.tempo, params.time_note)
            }
        }

        Note::Freq { frequency, duration, velocity } => {
            let dur_secs = params.metric_to_seconds(duration.value());
            let vel_byte = (velocity * params.velocity * 127.0).clamp(0.0, 127.0) as u8;
            let gate_dur = dur_secs * params.gate;

            // Find nearest MIDI note and calculate pitch bend compensation
            let (midi_note, bend_value) = freq_to_midi_with_bend(*frequency, 48.0);
            let transposed_midi = (midi_note as i16 + params.transpose as i16).clamp(0, 127) as u8;

            // Set pitch bend first
            co.yield_(MidiEvent {
                time: start_time + time_offset,
                channel: params.channel,
                event_type: MidiEventType::PitchBend { value: bend_value },
            }).await;

            // Note on
            co.yield_(MidiEvent {
                time: start_time + time_offset,
                channel: params.channel,
                event_type: MidiEventType::NoteOn { note: transposed_midi, velocity: vel_byte },
            }).await;

            // Note off
            co.yield_(MidiEvent {
                time: start_time + time_offset + gate_dur,
                channel: params.channel,
                event_type: MidiEventType::NoteOff { note: transposed_midi },
            }).await;

            // Reset pitch bend
            co.yield_(MidiEvent {
                time: start_time + time_offset + gate_dur + 0.001,
                channel: params.channel,
                event_type: MidiEventType::PitchBend { value: 8192 },
            }).await;

            params.last_duration = *duration;
            start_time + dur_secs
        }

        Note::Sweep { start_freq, end_freq, duration, velocity } => {
            let dur_secs = params.metric_to_seconds(duration.value());
            let vel_byte = (velocity * params.velocity * 127.0).clamp(0.0, 127.0) as u8;
            let gate_dur = dur_secs * params.gate;

            // Find nearest MIDI note to start frequency
            let (midi_note, initial_bend) = freq_to_midi_with_bend(*start_freq, 48.0);
            let transposed_midi = (midi_note as i16 + params.transpose as i16).clamp(0, 127) as u8;
            let base_freq = midi_to_freq(midi_note);

            // Set initial pitch bend
            co.yield_(MidiEvent {
                time: start_time + time_offset,
                channel: params.channel,
                event_type: MidiEventType::PitchBend { value: initial_bend },
            }).await;

            // Note on
            co.yield_(MidiEvent {
                time: start_time + time_offset,
                channel: params.channel,
                event_type: MidiEventType::NoteOn { note: transposed_midi, velocity: vel_byte },
            }).await;

            // Generate pitch bend events for sweep (30 per second)
            let steps = (dur_secs * 30.0) as usize;
            for step in 1..=steps {
                let progress = step as f32 / steps as f32;
                // Exponential interpolation for natural-sounding sweep
                let current_freq = start_freq * (end_freq / start_freq).powf(progress);
                let semitones_from_base = 12.0 * (current_freq / base_freq).log2();
                let bend_value = semitones_to_pitch_bend(semitones_from_base, 48.0);

                co.yield_(MidiEvent {
                    time: start_time + time_offset + (progress * dur_secs),
                    channel: params.channel,
                    event_type: MidiEventType::PitchBend { value: bend_value },
                }).await;
            }

            // Note off
            co.yield_(MidiEvent {
                time: start_time + time_offset + gate_dur,
                channel: params.channel,
                event_type: MidiEventType::NoteOff { note: transposed_midi },
            }).await;

            // Reset pitch bend
            co.yield_(MidiEvent {
                time: start_time + time_offset + gate_dur + 0.001,
                channel: params.channel,
                event_type: MidiEventType::PitchBend { value: 8192 },
            }).await;

            params.last_duration = *duration;
            start_time + dur_secs
        }
    }
}

// =============================================================================
// BATCH PROCESSING (uses streaming internally)
// =============================================================================

/// Convert a Note tree to a list of MIDI events (batch mode)
/// This collects all events from the streaming iterator and sorts them
pub fn note_to_events(note: &Note, tempo: f32) -> Vec<MidiEvent> {
    let mut events: Vec<MidiEvent> = note.event_stream(tempo).collect();
    events.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    events
}

// =============================================================================
// MIDI FILE GENERATION
// =============================================================================

/// Generate MIDI file bytes from events
pub fn events_to_midi(events: &[MidiEvent]) -> Vec<u8> {
    let ticks_per_quarter = 480u16;
    let tempo_us = 500000u32; // 120 BPM default

    let mut data = Vec::new();

    // MIDI Header
    data.extend_from_slice(b"MThd");
    data.extend_from_slice(&6u32.to_be_bytes());
    data.extend_from_slice(&0u16.to_be_bytes()); // Format 0
    data.extend_from_slice(&1u16.to_be_bytes()); // 1 track
    data.extend_from_slice(&ticks_per_quarter.to_be_bytes());

    let mut track = Vec::new();

    // Tempo meta event
    track.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03]);
    track.push((tempo_us >> 16) as u8);
    track.push((tempo_us >> 8) as u8);
    track.push(tempo_us as u8);

    let mut last_tick = 0u32;
    let ticks_per_sec = (ticks_per_quarter as f32) * 2.0;

    for event in events {
        // Skip comments in MIDI output
        if matches!(event.event_type, MidiEventType::Comment(_)) {
            continue;
        }

        let tick = (event.time * ticks_per_sec) as u32;
        let delta = tick.saturating_sub(last_tick);
        last_tick = tick;

        write_variable_length(&mut track, delta);

        match &event.event_type {
            MidiEventType::NoteOn { note, velocity } => {
                track.push(0x90 | event.channel);
                track.push(*note);
                track.push(*velocity);
            }
            MidiEventType::NoteOff { note } => {
                track.push(0x80 | event.channel);
                track.push(*note);
                track.push(0);
            }
            MidiEventType::ProgramChange { program } => {
                track.push(0xC0 | event.channel);
                track.push(*program);
            }
            MidiEventType::ControlChange { controller, value } => {
                track.push(0xB0 | event.channel);
                track.push(*controller);
                track.push(*value);
            }
            MidiEventType::PitchBend { value } => {
                // Pitch bend is split into LSB (lower 7 bits) and MSB (upper 7 bits)
                let lsb = (*value & 0x7F) as u8;
                let msb = ((*value >> 7) & 0x7F) as u8;
                track.push(0xE0 | event.channel);
                track.push(lsb);
                track.push(msb);
            }
            MidiEventType::Comment(_) => {}
        }
    }

    track.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

    data.extend_from_slice(b"MTrk");
    data.extend_from_slice(&(track.len() as u32).to_be_bytes());
    data.extend_from_slice(&track);

    data
}

fn write_variable_length(data: &mut Vec<u8>, mut value: u32) {
    if value == 0 {
        data.push(0);
        return;
    }

    let mut bytes = Vec::new();
    while value > 0 {
        bytes.push((value & 0x7F) as u8);
        value >>= 7;
    }
    bytes.reverse();

    for (idx, byte) in bytes.iter().enumerate() {
        if idx < bytes.len() - 1 {
            data.push(byte | 0x80);
        } else {
            data.push(*byte);
        }
    }
}

/// Generate MIDI from a Note composition
pub fn compose_to_midi(composition: &Note, tempo: f32) -> Vec<u8> {
    let events = note_to_events(composition, tempo);
    events_to_midi(&events)
}

// =============================================================================
// PRELUDE - convenient imports
// =============================================================================

pub mod prelude {
    pub use crate::{
        // Core types
        Note, Duration, ParamValue, Params,
        MidiEvent, MidiEventType,
        // Envelope types
        TimeUnit, EnvelopeDuration, EnvelopePoint, InterpolationType, Envelope,
        // Composition functions
        ser, par, parmin, forkseq, forkpar,
        rep, repeat, comment,
        // Frequency/sweep functions
        freq, freq_vel, sweep, sweep_vel,
        // Parameter functions
        tempo, channel, program, velocity, gate, transpose, key, time_note, GM,
        set_macro, clear_macro,
        // Synthesis parameters
        patch, duty, noise_type, instrument, volume,
        // ADSR envelope parameters
        attack_envelope, sustain_envelope, release_envelope, release_duration,
        envelope_preset,
        // Envelope helpers
        parse_envelope_point, param_value_from, ParamValueConvertible,
        // Key signature
        parse_key_signature, get_key_adjustment,
        // Frequency/pitch bend helpers
        midi_to_freq, freq_to_midi_with_bend, semitones_to_pitch_bend,
        // MIDI generation
        compose_to_midi, note_to_events, events_to_midi,
        // Generator support
        yield_,
    };

    // Re-export macros (only ones not already exported above)
    pub use crate::{
        // Additional composition macro
        forkser,
        // Envelope macros
        env, linen, cosen, expen, cuben,
        // Parameter macro
        param,
        // Note with params macro
        note,
        // Generator macro
        music_gen,
    };

    // Re-export genawaiter for advanced usage
    pub use genawaiter;

    // Re-export synth module
    pub use crate::synth::{
        BasicSynth, Voice, SynthParams,
        Waveform, NoiseType, ADSREnvelope,
        render_to_samples, render_to_stereo_samples,
        SAMPLE_RATE,
    };

    // Re-export sfx module
    pub use crate::sfx::SfxLibrary;

    // Note: For note constants (c4q, d4h, etc.), rests (rq, rh, etc.),
    // duration ties (w, h, q, etc.), and previous pitch (p, pq, ph, etc.),
    // use `use music::*` in a local scope or import specific notes
}

// =============================================================================
// COMPOSITIONS
// =============================================================================

pub mod compositions {
    use crate::*;

    /// Upbeat dungeon exploration music - Dragon Quest inspired adventure in D minor
    pub fn dungeon_ambient() -> Note {
        // Driving drum beat - kick and snare pattern
        let drums = ser![
            program(GM::AcousticGrandPiano), // Ignored on drum channel
            channel(9), // Percussion channel
            velocity(0.9),
            rep(8, ser![
                // Kick on 1 and 3, snare on 2 and 4, hi-hat throughout
                par![c2e, fs2e],  // Kick + closed hi-hat
                fs2e,              // Hi-hat
                par![d2e, fs2e],  // Snare + hi-hat
                fs2e,              // Hi-hat
                par![c2e, fs2e],  // Kick + hi-hat
                fs2e,              // Hi-hat
                par![d2e, fs2e],  // Snare + hi-hat
                as2e,              // Open hi-hat
            ]),
        ];

        // Punchy arpeggiated bass line
        let bass = ser![
            program(GM::SynthBass1),
            velocity(0.85),
            rep(2, ser![
                // Dm
                d2e, d2e, d3e, d2e, a2e, d2e, d3e, a2e,
                // Am
                a1e, a1e, a2e, a1e, e2e, a1e, a2e, e2e,
                // Bb
                bf1e, bf1e, bf2e, bf1e, f2e, bf1e, bf2e, f2e,
                // A (tension)
                a1e, a1e, a2e, a1e, e2e, a1e, cs2e, e2e,
            ]),
        ];

        // Adventurous melody - DQ style heroic theme
        let melody = ser![
            program(GM::SquareLead),
            channel(1),
            velocity(0.8),
            // First phrase
            d5q, f5e, e5e, d5q, a4q,
            bf4q, a4e, g4e, a4h,
            // Second phrase
            d5q, f5e, g5e, a5h,
            g5q, f5e, e5e, d5h,
            // Third phrase - higher energy
            a5q, a5e, g5e, f5q, e5q,
            d5q, e5e, f5e, e5q, d5q,
            // Resolution
            a4q, bf4e, a4e, g4q, a4q,
            d5w,
        ];

        // Counter melody - arpeggiated chords
        let counter = ser![
            program(GM::SawtoothLead),
            channel(2),
            velocity(0.5),
            rep(2, ser![
                // Dm arpeggio
                d4i, f4i, a4i, f4i, d4i, f4i, a4i, f4i,
                d4i, f4i, a4i, f4i, d4i, f4i, a4i, f4i,
                // Am arpeggio
                a3i, c4i, e4i, c4i, a3i, c4i, e4i, c4i,
                a3i, c4i, e4i, c4i, a3i, c4i, e4i, c4i,
                // Bb arpeggio
                bf3i, d4i, f4i, d4i, bf3i, d4i, f4i, d4i,
                bf3i, d4i, f4i, d4i, bf3i, d4i, f4i, d4i,
                // A arpeggio
                a3i, cs4i, e4i, cs4i, a3i, cs4i, e4i, cs4i,
                a3i, cs4i, e4i, cs4i, a3i, cs4i, e4i, cs4i,
            ]),
        ];

        ser![
            tempo(128.0),
            par![drums, bass, melody, counter],
        ]
    }

    /// Intense combat music - driving rhythm in E minor
    pub fn combat() -> Note {
        let drums = ser![
            program(GM::Timpani),
            channel(9),
            rep(4, ser![
                c2q, re, c2e,
                c2q, g2q,
            ]),
        ];

        let bass = ser![
            program(GM::ElectricBassFinger),
            rep(4, ser![
                e2e, e2e, e2e, e2e,
                e2e, g2e, e2e, b1e,
            ]),
        ];

        let melody = ser![
            program(GM::StringEnsemble1),
            channel(1),
            e4q, g4e, a4e, b4h,
            a4q, g4q, e4h,
            e4q, g4e, a4e, b4q, d5q,
            c5h, b4h,
        ];

        ser![
            tempo(140.0),
            par![drums, bass, melody],
        ]
    }

    /// Triumphant victory fanfare
    pub fn victory() -> Note {
        ser![
            tempo(120.0),
            program(GM::Trumpet),
            g4e, g4e, g4e, g4qd,
            ef4h,
            g4e, g4e, g4e, g4qd,
            c5w,
        ]
    }

    /// Somber game over music
    pub fn game_over() -> Note {
        ser![
            tempo(60.0),
            program(GM::StringEnsemble1),
            par![a3h, c4h, e4h],
            par![g3h, b3h, d4h],
            par![f3h, a3h, c4h],
            par![e3w, g3w, b3w],
        ]
    }

    /// Mysterious discovery jingle
    pub fn discovery() -> Note {
        ser![
            tempo(100.0),
            program(GM::OrchestralHarp),
            c4i, e4i, g4i, c5i,
            e5i, g5i, c6q,
        ]
    }

    /// Shop/safe room music - calm and pleasant
    pub fn shop() -> Note {
        let melody = ser![
            program(GM::Flute),
            c5q, e5q, g5q, e5q,
            f5q, a5q, g5h,
            e5q, g5q, c6h,
            b5q, a5q, g5h,
        ];

        let chords = ser![
            program(GM::AcousticGrandPiano),
            channel(1),
            par![c4h, e4h, g4h],
            par![f4h, a4h, c5h],
            par![c4h, e4h, g4h],
            par![g3h, b3h, d4h],
        ];

        ser![
            tempo(90.0),
            par![melody, chords],
        ]
    }

    /// Example using advanced features: forkseq, transpose, previous pitch
    pub fn advanced_demo() -> Note {
        ser![
            tempo(100.0),
            program(GM::AcousticGrandPiano),
            comment!("Main theme"),

            // Use forkseq to layer a sustained note under a melody
            forkseq![c3w],
            c4q, e4q, g4q, c5q,

            comment!("Transpose up a fifth"),
            transpose(7),
            c4q, e4q, g4q, c5q,

            comment!("Using previous pitch"),
            transpose(0),
            c4q, p, p, ph,  // C4 quarter, then three more C4s

            comment!("Using gate for staccato"),
            gate(0.3),
            c4e, d4e, e4e, f4e, g4e, a4e, b4e, c5e,

            gate(0.9),
            c5w,
        ]
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relative_envelope_parsing() {
        // Absolute values
        let point = parse_envelope_point("q1.0");
        assert_eq!(point.value, 1.0);
        assert!(!point.relative);

        // Relative positive
        let point = parse_envelope_point("q+0.5");
        assert_eq!(point.value, 0.5);
        assert!(point.relative);

        // Relative negative
        let point = parse_envelope_point("q-0.3");
        assert_eq!(point.value, -0.3);
        assert!(point.relative);

        // Explicit absolute (for negative values)
        let point = parse_envelope_point("q=-0.5");
        assert_eq!(point.value, -0.5);
        assert!(!point.relative);

        // Time-based with relative
        let point = parse_envelope_point("100ms+0.2");
        assert_eq!(point.value, 0.2);
        assert!(point.relative);
    }

    #[test]
    fn test_relative_envelope_evaluation() {
        // Test with seconds-based envelope for clearer timing
        // 0s -> 1.0, 1s -> +0.2 (=1.2), 2s -> -0.1 (=1.1)
        let test_envelope = Envelope::with_points("test", vec![
            EnvelopePoint::new_seconds(1.0, 0.0),           // absolute 1.0 at 0s
            EnvelopePoint::new_seconds_relative(0.2, 1.0),  // +0.2 = 1.2 at 1s
            EnvelopePoint::new_seconds_relative(-0.1, 1.0), // -0.1 = 1.1 at 2s
        ]);

        let tempo = 120.0;
        let time_note = 0.25;

        // At time 0: should be 1.0
        let val = test_envelope.evaluate(0.0, tempo, time_note, None);
        assert!((val - 1.0).abs() < 0.01, "Expected 1.0, got {}", val);

        // At time 0.5 (halfway to 1s): should be interpolated to 1.1
        let val = test_envelope.evaluate(0.5, tempo, time_note, None);
        assert!((val - 1.1).abs() < 0.01, "Expected 1.1, got {}", val);

        // At time 1.0: should be 1.2
        let val = test_envelope.evaluate(1.0, tempo, time_note, None);
        assert!((val - 1.2).abs() < 0.01, "Expected 1.2, got {}", val);

        // At time 1.5 (halfway to 2s): should be interpolated to 1.15
        let val = test_envelope.evaluate(1.5, tempo, time_note, None);
        assert!((val - 1.15).abs() < 0.01, "Expected 1.15, got {}", val);

        // At time 2.0: should be 1.1
        let val = test_envelope.evaluate(2.0, tempo, time_note, None);
        assert!((val - 1.1).abs() < 0.01, "Expected 1.1, got {}", val);
    }

    #[test]
    fn test_envelope_macro_with_relative() {
        // Test that env! macro parses relative values correctly
        // env! returns Note::Envelope, so extract the inner Envelope
        let note = env!(q1.0, q+0.5, q-0.2);
        if let Note::Envelope(test_env) = note {
            assert_eq!(test_env.points.len(), 3);
            assert!(!test_env.points[0].relative);
            assert!(test_env.points[1].relative);
            assert!(test_env.points[2].relative);
        } else {
            panic!("env! should return Note::Envelope");
        }
    }
}
