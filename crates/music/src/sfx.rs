//! Sound Effects Module
//!
//! Sound effects expressed as music DSL compositions, rendered to samples.

use crate::prelude::*;
use crate::synth::{BasicSynth, SynthParams, Waveform, NoiseType, ADSREnvelope};
use crate::{ser, par};
// Import note constants (c4q, a4i, etc.)
use crate::*;

/// Render a composition to audio samples using BasicSynth
fn render_composition(composition: &Note, params: SynthParams) -> Vec<f32> {
    let events = note_to_events(composition, 120.0); // tempo doesn't matter, we use absolute time

    // Find total duration from events
    let duration = events.iter()
        .map(|ev| ev.time)
        .fold(0.0_f32, |a, b| a.max(b)) + 0.1; // Add small tail

    let total_samples = (duration * SAMPLE_RATE as f32) as usize;
    let mut synth = BasicSynth::new();
    synth.master_volume = 0.7;

    // Apply params
    synth.set_waveform(params.waveform);
    synth.set_duty(params.duty);
    synth.set_noise_type(params.noise_type);
    synth.set_envelope(params.envelope);

    let mut samples = Vec::with_capacity(total_samples);
    let mut event_idx = 0;

    for _ in 0..total_samples {
        while event_idx < events.len() && events[event_idx].time <= synth.current_time() {
            synth.process_event(&events[event_idx]);
            event_idx += 1;
        }
        samples.push(synth.generate_sample());
    }

    samples
}

/// Render composition with multiple synth tracks (for layered sounds)
fn render_layered(layers: &[(Note, SynthParams)]) -> Vec<f32> {
    let mut result: Vec<f32> = Vec::new();

    for (composition, params) in layers {
        let samples = render_composition(composition, params.clone());

        if result.is_empty() {
            result = samples;
        } else {
            // Mix
            let len = result.len().max(samples.len());
            result.resize(len, 0.0);
            for (idx, &sample) in samples.iter().enumerate() {
                if idx < result.len() {
                    result[idx] += sample;
                }
            }
        }
    }

    result
}

// =============================================================================
// SYNTH PARAMETER PRESETS
// =============================================================================

fn noise_params(duration: f32) -> SynthParams {
    SynthParams {
        waveform: Waveform::Noise,
        noise_type: NoiseType::White,
        envelope: ADSREnvelope {
            attack_time: 0.001,
            decay_time: duration - 0.001,
            sustain_level: 0.0,
            release_time: 0.01,
        },
        volume: 0.5,
        duty: 0.5,
    }
}

fn square_params(duration: f32) -> SynthParams {
    SynthParams {
        waveform: Waveform::Square,
        envelope: ADSREnvelope {
            attack_time: 0.001,
            decay_time: duration - 0.001,
            sustain_level: 0.0,
            release_time: 0.01,
        },
        volume: 0.5,
        duty: 0.5,
        noise_type: NoiseType::White,
    }
}

fn triangle_params(duration: f32) -> SynthParams {
    SynthParams {
        waveform: Waveform::Triangle,
        envelope: ADSREnvelope {
            attack_time: 0.005,
            decay_time: duration - 0.01,
            sustain_level: 0.3,
            release_time: 0.02,
        },
        volume: 0.5,
        duty: 0.5,
        noise_type: NoiseType::White,
    }
}

// =============================================================================
// SOUND EFFECT COMPOSITIONS
// =============================================================================

/// Sword slash sound: High-to-low sweep ending in a satisfying crunch/thud
pub fn hit_composition() -> Note {
    ser![
        tempo(400.0), // Quarter = 0.15s
        par![
            // Main sweep - periodic noise high to low
            ser![
                patch("noise"),
                noise_type("periodic"),
                sweep(6000.0, 150.0, Duration::Quarter),
            ],
            // Ending thud - low square hit delayed slightly
            ser![
                rq, // Rest for half the duration
                patch("square"),
                freq(80.0, Duration::Eighth),
            ],
            // Crunch layer - white noise burst at the end
            ser![
                rq,
                patch("noise"),
                freq(100.0, Duration::Eighth),
            ],
        ]
    ]
}

fn sweep_noise_params() -> SynthParams {
    SynthParams {
        waveform: Waveform::Noise,
        noise_type: NoiseType::Periodic,
        envelope: ADSREnvelope {
            attack_time: 0.002,
            decay_time: 0.12,
            sustain_level: 0.1,
            release_time: 0.02,
        },
        volume: 0.6,
        duty: 0.5,
    }
}

fn thud_params() -> SynthParams {
    SynthParams {
        waveform: Waveform::Square,
        envelope: ADSREnvelope {
            attack_time: 0.001,
            decay_time: 0.06,
            sustain_level: 0.0,
            release_time: 0.02,
        },
        volume: 0.5,
        duty: 0.5,
        noise_type: NoiseType::White,
    }
}

fn crunch_params() -> SynthParams {
    SynthParams {
        waveform: Waveform::Noise,
        noise_type: NoiseType::White,
        envelope: ADSREnvelope {
            attack_time: 0.001,
            decay_time: 0.04,
            sustain_level: 0.0,
            release_time: 0.01,
        },
        volume: 0.4,
        duty: 0.5,
    }
}

/// Hit sound rendered to samples
pub fn hit() -> Vec<f32> {
    render_layered(&[
        // Periodic noise sweep high→low (the main "SHHWIK")
        (ser![tempo(400.0), sweep(6000.0, 150.0, Duration::Quarter)], sweep_noise_params()),
        // Low thud at the end
        (ser![tempo(400.0), re, freq(80.0, Duration::Eighth)], thud_params()),
        // White noise crunch at the end
        (ser![tempo(400.0), re, freq(100.0, Duration::Eighth)], crunch_params()),
    ])
}

/// Pickup sound: Rising arpeggio A4→C#5→E5→A5 (triangle waves)
pub fn pickup_composition() -> Note {
    // Total duration ~0.25s, 4 notes at 0.04s spacing, each ~0.08s
    // At tempo 750, sixteenth = 0.04s, eighth = 0.08s
    ser![
        tempo(750.0),
        patch("triangle"),
        a4i, cs5i, e5i, a5i,
    ]
}

/// Pickup sound rendered to samples
pub fn pickup() -> Vec<f32> {
    let composition = ser![
        tempo(750.0),
        a4i, cs5i, e5i, a5i,
    ];
    render_composition(&composition, triangle_params(0.08))
}

/// Hurt sound: Pitch sweep 300Hz→80Hz over 0.15s
pub fn hurt_composition() -> Note {
    // At tempo 400, quarter = 0.15s
    ser![
        tempo(400.0),
        patch("square"),
        sweep(300.0, 80.0, Duration::Quarter),
    ]
}

/// Hurt sound rendered to samples
pub fn hurt() -> Vec<f32> {
    let composition = ser![
        tempo(400.0),
        sweep(300.0, 80.0, Duration::Quarter),
    ];
    render_composition(&composition, square_params(0.15))
}

/// Step sound: Very quiet short noise burst (0.03s)
pub fn step_composition() -> Note {
    // At tempo 2000, quarter = 0.03s
    ser![
        tempo(2000.0),
        patch("noise"),
        velocity(0.4),
        c4q,
    ]
}

/// Step sound rendered to samples
pub fn step() -> Vec<f32> {
    let composition = ser![
        tempo(2000.0),
        velocity(0.4),
        c4q,
    ];
    let mut params = noise_params(0.03);
    params.volume = 0.2;
    render_composition(&composition, params)
}

/// Enemy death sound: Pitch sweep 400Hz→50Hz + noise over 0.25s
pub fn enemy_die_composition() -> Note {
    // At tempo 240, quarter = 0.25s
    ser![
        tempo(240.0),
        par![
            ser![
                patch("square"),
                sweep(400.0, 50.0, Duration::Quarter),
            ],
            ser![
                patch("noise"),
                c4q,
            ],
        ]
    ]
}

/// Enemy death sound rendered to samples
pub fn enemy_die() -> Vec<f32> {
    render_layered(&[
        (ser![tempo(240.0), sweep(400.0, 50.0, Duration::Quarter)], square_params(0.25)),
        (ser![tempo(240.0), c4q], noise_params(0.25)),
    ])
}

/// Stairs sound: Rising C major arpeggio C4→E4→G4→C5→E5→G5 (triangle)
pub fn stairs_composition() -> Note {
    // Total ~0.5s, 6 notes, ~0.07s spacing, ~0.12s each
    // At tempo ~428, sixteenth = 0.07s
    ser![
        tempo(428.0),
        patch("triangle"),
        c4i, e4i, g4i, c5i, e5i, g5i,
    ]
}

/// Stairs sound rendered to samples
pub fn stairs() -> Vec<f32> {
    let composition = ser![
        tempo(428.0),
        c4i, e4i, g4i, c5i, e5i, g5i,
    ];
    let mut params = triangle_params(0.12);
    params.volume = 0.4;
    render_composition(&composition, params)
}

// =============================================================================
// SFX LIBRARY
// =============================================================================

/// Pre-generated sound effect library
pub struct SfxLibrary {
    pub hit: Vec<f32>,
    pub pickup: Vec<f32>,
    pub hurt: Vec<f32>,
    pub step: Vec<f32>,
    pub enemy_die: Vec<f32>,
    pub stairs: Vec<f32>,
}

impl SfxLibrary {
    /// Generate all sound effects
    pub fn new() -> Self {
        Self {
            hit: hit(),
            pickup: pickup(),
            hurt: hurt(),
            step: step(),
            enemy_die: enemy_die(),
            stairs: stairs(),
        }
    }
}

impl Default for SfxLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hit_sound() {
        let samples = hit();
        assert!(!samples.is_empty());
        assert!(samples.iter().any(|&s| s.abs() > 0.01));
    }

    #[test]
    fn test_all_sfx() {
        let lib = SfxLibrary::new();
        assert!(!lib.hit.is_empty());
        assert!(!lib.pickup.is_empty());
        assert!(!lib.hurt.is_empty());
        assert!(!lib.step.is_empty());
        assert!(!lib.enemy_die.is_empty());
        assert!(!lib.stairs.is_empty());
    }

    #[test]
    fn test_sweep_generates_events() {
        let composition = sweep(300.0, 80.0, Duration::Quarter);
        let events = note_to_events(&composition, 120.0);
        // Should have: pitch bend, note on, multiple pitch bends, note off, pitch bend reset
        assert!(events.len() > 5);
    }
}
