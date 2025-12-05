//! BasicSynth - A software synthesizer that processes MIDI commands
//!
//! This module provides audio synthesis capabilities including:
//! - Multiple waveforms: sine, square, triangle, sawtooth, noise
//! - ADSR envelope control
//! - Duty cycle for square waves
//! - Multiple noise types: white, brown, pink, periodic

use std::collections::HashMap;
use std::f32::consts::PI;

use crate::{MidiEvent, MidiEventType, Params};

/// Sample rate for audio generation (standard CD quality)
pub const SAMPLE_RATE: u32 = 44100;

/// Waveform types supported by the synthesizer
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Waveform {
    Sine,
    Square,
    Triangle,
    Sawtooth,
    Noise,
}

impl Default for Waveform {
    fn default() -> Self {
        Waveform::Square
    }
}

impl From<&str> for Waveform {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "sine" => Waveform::Sine,
            "square" => Waveform::Square,
            "triangle" => Waveform::Triangle,
            "sawtooth" | "saw" => Waveform::Sawtooth,
            "noise" => Waveform::Noise,
            _ => Waveform::Square,
        }
    }
}

/// Noise types for noise waveform
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoiseType {
    White,
    Brown,
    Pink,
    Periodic,
}

impl Default for NoiseType {
    fn default() -> Self {
        NoiseType::White
    }
}

impl From<&str> for NoiseType {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "white" => NoiseType::White,
            "brown" | "red" => NoiseType::Brown,
            "pink" => NoiseType::Pink,
            "periodic" => NoiseType::Periodic,
            _ => NoiseType::White,
        }
    }
}

/// ADSR envelope state
#[derive(Clone, Debug)]
pub struct ADSREnvelope {
    pub attack_time: f32,   // seconds
    pub decay_time: f32,    // seconds
    pub sustain_level: f32, // 0.0 - 1.0
    pub release_time: f32,  // seconds
}

impl Default for ADSREnvelope {
    fn default() -> Self {
        Self {
            attack_time: 0.01,
            decay_time: 0.1,
            sustain_level: 0.7,
            release_time: 0.1,
        }
    }
}

impl ADSREnvelope {
    /// Create a percussion envelope (fast attack, quick decay, no sustain)
    pub fn percussion() -> Self {
        Self {
            attack_time: 0.001,
            decay_time: 0.15,
            sustain_level: 0.0,
            release_time: 0.05,
        }
    }

    /// Get envelope amplitude at a given time
    /// `note_on_time` is when the note started
    /// `note_off_time` is when the note was released (None if still held)
    pub fn amplitude(&self, time: f32, note_on_time: f32, note_off_time: Option<f32>) -> f32 {
        let time_since_on = time - note_on_time;

        if let Some(off_time) = note_off_time {
            // In release phase
            let time_since_off = time - off_time;
            if time_since_off >= self.release_time {
                return 0.0;
            }

            // Get amplitude at note-off time
            let amp_at_off = self.amplitude_at_time(off_time - note_on_time);
            // Linear decay from that amplitude
            amp_at_off * (1.0 - time_since_off / self.release_time)
        } else {
            self.amplitude_at_time(time_since_on)
        }
    }

    fn amplitude_at_time(&self, time: f32) -> f32 {
        if time < 0.0 {
            return 0.0;
        }

        if time < self.attack_time {
            // Attack phase - linear rise
            time / self.attack_time
        } else if time < self.attack_time + self.decay_time {
            // Decay phase - linear fall to sustain
            let decay_progress = (time - self.attack_time) / self.decay_time;
            1.0 - decay_progress * (1.0 - self.sustain_level)
        } else {
            // Sustain phase
            self.sustain_level
        }
    }
}

/// Pitch envelope for frequency sweeps
#[derive(Clone, Debug)]
pub struct PitchEnvelope {
    pub start_freq: f32,
    pub end_freq: f32,
    pub duration: f32,
}

impl PitchEnvelope {
    pub fn new(start_freq: f32, end_freq: f32, duration: f32) -> Self {
        Self { start_freq, end_freq, duration }
    }

    /// Get frequency at a given time (linear interpolation)
    pub fn frequency_at(&self, time_since_start: f32) -> f32 {
        if time_since_start <= 0.0 {
            self.start_freq
        } else if time_since_start >= self.duration {
            self.end_freq
        } else {
            let progress = time_since_start / self.duration;
            self.start_freq + (self.end_freq - self.start_freq) * progress
        }
    }

    /// Get frequency with exponential interpolation (sounds more natural for pitch)
    pub fn frequency_at_exp(&self, time_since_start: f32) -> f32 {
        if time_since_start <= 0.0 {
            self.start_freq
        } else if time_since_start >= self.duration {
            self.end_freq
        } else {
            let progress = time_since_start / self.duration;
            // Exponential interpolation in frequency space
            self.start_freq * (self.end_freq / self.start_freq).powf(progress)
        }
    }
}

/// A single voice in the synthesizer
#[derive(Clone)]
pub struct Voice {
    pub midi_note: u8,
    pub velocity: f32,
    pub frequency: f32,
    pub phase: f32,
    pub waveform: Waveform,
    pub duty: f32,
    pub noise_type: NoiseType,
    pub envelope: ADSREnvelope,
    pub start_time: f32,
    pub release_time: Option<f32>,
    pub volume: f32,
    // Noise state
    pub noise_state: f32,
    pub lfsr: u16, // For periodic noise
    // Pitch envelope for sweeps
    pub pitch_envelope: Option<PitchEnvelope>,
}

impl Voice {
    pub fn new(midi_note: u8, velocity: f32, params: &SynthParams, start_time: f32) -> Self {
        Self {
            midi_note,
            velocity,
            frequency: midi_to_freq(midi_note),
            phase: 0.0,
            waveform: params.waveform,
            duty: params.duty,
            noise_type: params.noise_type,
            envelope: params.envelope.clone(),
            start_time,
            release_time: None,
            volume: params.volume,
            noise_state: 0.0,
            lfsr: 0x7FFF, // LFSR seed for periodic noise
            pitch_envelope: None,
        }
    }

    /// Create a voice with a pitch sweep
    pub fn with_pitch_sweep(
        start_freq: f32,
        end_freq: f32,
        duration: f32,
        velocity: f32,
        params: &SynthParams,
        start_time: f32,
    ) -> Self {
        Self {
            midi_note: 60, // Doesn't matter for sweep
            velocity,
            frequency: start_freq,
            phase: 0.0,
            waveform: params.waveform,
            duty: params.duty,
            noise_type: params.noise_type,
            envelope: params.envelope.clone(),
            start_time,
            release_time: None,
            volume: params.volume,
            noise_state: 0.0,
            lfsr: 0x7FFF,
            pitch_envelope: Some(PitchEnvelope::new(start_freq, end_freq, duration)),
        }
    }

    /// Generate a sample for this voice
    /// pitch_bend_semitones: global pitch bend from the synth (in semitones)
    pub fn generate_sample(&mut self, current_time: f32, pitch_bend_semitones: f32) -> f32 {
        let env_amp = self.envelope.amplitude(current_time, self.start_time, self.release_time);

        if env_amp <= 0.0001 {
            return 0.0;
        }

        // Apply pitch envelope if present, otherwise use base frequency
        let base_freq = if let Some(ref pitch_env) = self.pitch_envelope {
            let time_since_start = current_time - self.start_time;
            pitch_env.frequency_at_exp(time_since_start)
        } else {
            self.frequency
        };

        // Apply global pitch bend (in semitones)
        let freq = base_freq * 2.0_f32.powf(pitch_bend_semitones / 12.0);

        let sample = match self.waveform {
            Waveform::Sine => self.generate_sine(),
            Waveform::Square => self.generate_square(),
            Waveform::Triangle => self.generate_triangle(),
            Waveform::Sawtooth => self.generate_sawtooth(),
            Waveform::Noise => self.generate_noise(),
        };

        // Advance phase using current frequency (may be modulated by pitch envelope and bend)
        self.phase += freq / SAMPLE_RATE as f32;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        sample * env_amp * self.velocity * self.volume
    }

    fn generate_sine(&self) -> f32 {
        (self.phase * 2.0 * PI).sin()
    }

    fn generate_square(&self) -> f32 {
        if self.phase < self.duty {
            1.0
        } else {
            -1.0
        }
    }

    fn generate_triangle(&self) -> f32 {
        if self.phase < 0.5 {
            4.0 * self.phase - 1.0
        } else {
            3.0 - 4.0 * self.phase
        }
    }

    fn generate_sawtooth(&self) -> f32 {
        2.0 * self.phase - 1.0
    }

    fn generate_noise(&mut self) -> f32 {
        match self.noise_type {
            NoiseType::White => {
                // Simple white noise
                rand_f32() * 2.0 - 1.0
            }
            NoiseType::Brown => {
                // Brown noise - integrate white noise
                let white = rand_f32() * 2.0 - 1.0;
                self.noise_state += white * 0.02;
                self.noise_state = self.noise_state.clamp(-1.0, 1.0);
                self.noise_state
            }
            NoiseType::Pink => {
                // Simplified pink noise approximation
                let white = rand_f32() * 2.0 - 1.0;
                self.noise_state = 0.99 * self.noise_state + 0.01 * white;
                (self.noise_state + white * 0.2).clamp(-1.0, 1.0)
            }
            NoiseType::Periodic => {
                // LFSR-based periodic noise (NES style)
                let bit = ((self.lfsr >> 0) ^ (self.lfsr >> 1)) & 1;
                self.lfsr = (self.lfsr >> 1) | (bit << 14);
                if self.lfsr & 1 == 1 { 1.0 } else { -1.0 }
            }
        }
    }

    /// Check if this voice has finished (envelope complete)
    pub fn is_finished(&self, current_time: f32) -> bool {
        if let Some(release_time) = self.release_time {
            current_time >= release_time + self.envelope.release_time
        } else {
            false
        }
    }

    /// Release this voice
    pub fn release(&mut self, time: f32) {
        self.release_time = Some(time);
    }
}

/// Synthesis parameters (from Params)
#[derive(Clone)]
pub struct SynthParams {
    pub waveform: Waveform,
    pub duty: f32,
    pub noise_type: NoiseType,
    pub volume: f32,
    pub envelope: ADSREnvelope,
}

impl Default for SynthParams {
    fn default() -> Self {
        Self {
            waveform: Waveform::Square,
            duty: 0.5,
            noise_type: NoiseType::White,
            volume: 0.5,
            envelope: ADSREnvelope::default(),
        }
    }
}

impl From<&Params> for SynthParams {
    fn from(params: &Params) -> Self {
        let mut synth_params = SynthParams::default();
        synth_params.waveform = Waveform::from(params.patch.as_str());
        synth_params.duty = params.duty;
        synth_params.noise_type = NoiseType::from(params.noise_type.as_str());
        synth_params.volume = params.volume * params.velocity;

        // Apply envelope preset if set
        if let Some(ref preset) = params.envelope_preset {
            if preset == "percussion" {
                synth_params.envelope = ADSREnvelope::percussion();
            }
        }

        // TODO: Apply custom envelopes from attack_envelope, sustain_envelope, release_envelope

        synth_params
    }
}

/// The main synthesizer
pub struct BasicSynth {
    voices: Vec<Voice>,
    params: SynthParams,
    current_time: f32,
    sample_rate: u32,
    /// Map of active note -> voice index for note-off handling
    note_to_voice: HashMap<u8, usize>,
    /// Master volume
    pub master_volume: f32,
    /// Pitch bend in semitones (applied to all voices)
    pub pitch_bend: f32,
    /// Pitch bend range in semitones (default 48 for wide SFX sweeps)
    pub pitch_bend_range: f32,
}

impl Default for BasicSynth {
    fn default() -> Self {
        Self::new()
    }
}

impl BasicSynth {
    pub fn new() -> Self {
        Self {
            voices: Vec::new(),
            params: SynthParams::default(),
            current_time: 0.0,
            sample_rate: SAMPLE_RATE,
            note_to_voice: HashMap::new(),
            master_volume: 0.5,
            pitch_bend: 0.0,
            pitch_bend_range: 48.0, // 4 octaves - wide range for SFX sweeps
        }
    }

    /// Create with custom sample rate
    pub fn with_sample_rate(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            ..Self::new()
        }
    }

    /// Process a MIDI event
    pub fn process_event(&mut self, event: &MidiEvent) {
        match &event.event_type {
            MidiEventType::NoteOn { note, velocity } => {
                self.note_on(*note, *velocity as f32 / 127.0);
            }
            MidiEventType::NoteOff { note } => {
                self.note_off(*note);
            }
            MidiEventType::ProgramChange { program: _ } => {
                // Could map programs to different patches
            }
            MidiEventType::ControlChange { controller, value } => {
                match controller {
                    7 => {
                        // Volume CC
                        self.params.volume = *value as f32 / 127.0;
                    }
                    1 => {
                        // Modulation wheel - could affect duty cycle
                        self.params.duty = 0.1 + (*value as f32 / 127.0) * 0.8;
                    }
                    _ => {}
                }
            }
            MidiEventType::PitchBend { value } => {
                // Convert 14-bit pitch bend to semitones using configured range
                let normalized = (*value as f32 - 8192.0) / 8192.0;
                self.pitch_bend = normalized * self.pitch_bend_range;
            }
            MidiEventType::Comment(_) => {}
            MidiEventType::ChannelEffects { .. } => {
                // Effects are handled externally via DSP processing
            }
        }
    }

    /// Process all events up to a given time
    pub fn process_events_until(&mut self, events: &[MidiEvent], until_time: f32) {
        for event in events {
            if event.time <= until_time {
                self.process_event(event);
            }
        }
    }

    /// Trigger a note on
    pub fn note_on(&mut self, midi_note: u8, velocity: f32) {
        // If this note is already playing, release it first
        if let Some(&voice_idx) = self.note_to_voice.get(&midi_note) {
            if voice_idx < self.voices.len() {
                self.voices[voice_idx].release(self.current_time);
            }
        }

        let voice = Voice::new(midi_note, velocity, &self.params, self.current_time);
        self.voices.push(voice);
        self.note_to_voice.insert(midi_note, self.voices.len() - 1);
    }

    /// Trigger a note off
    pub fn note_off(&mut self, midi_note: u8) {
        if let Some(&voice_idx) = self.note_to_voice.get(&midi_note) {
            if voice_idx < self.voices.len() {
                self.voices[voice_idx].release(self.current_time);
            }
        }
        self.note_to_voice.remove(&midi_note);
    }

    /// Generate a single sample (mono)
    pub fn generate_sample(&mut self) -> f32 {
        let mut output = 0.0;
        let pitch_bend = self.pitch_bend;

        for voice in &mut self.voices {
            output += voice.generate_sample(self.current_time, pitch_bend);
        }

        // Remove finished voices
        self.voices.retain(|v| !v.is_finished(self.current_time));

        // Advance time
        self.current_time += 1.0 / self.sample_rate as f32;

        // Apply master volume and soft clipping
        soft_clip(output * self.master_volume)
    }

    /// Generate stereo samples
    pub fn generate_stereo_sample(&mut self) -> (f32, f32) {
        let mono = self.generate_sample();
        (mono, mono) // Simple mono to stereo
    }

    /// Generate a buffer of samples
    pub fn generate_samples(&mut self, num_samples: usize) -> Vec<f32> {
        let mut buffer = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            buffer.push(self.generate_sample());
        }
        buffer
    }

    /// Generate a buffer of stereo interleaved samples
    pub fn generate_stereo_samples(&mut self, num_samples: usize) -> Vec<f32> {
        let mut buffer = Vec::with_capacity(num_samples * 2);
        for _ in 0..num_samples {
            let (left, right) = self.generate_stereo_sample();
            buffer.push(left);
            buffer.push(right);
        }
        buffer
    }

    /// Set the waveform
    pub fn set_waveform(&mut self, waveform: Waveform) {
        self.params.waveform = waveform;
    }

    /// Set duty cycle (for square wave)
    pub fn set_duty(&mut self, duty: f32) {
        self.params.duty = duty.clamp(0.0, 1.0);
    }

    /// Set noise type
    pub fn set_noise_type(&mut self, noise_type: NoiseType) {
        self.params.noise_type = noise_type;
    }

    /// Set the ADSR envelope
    pub fn set_envelope(&mut self, envelope: ADSREnvelope) {
        self.params.envelope = envelope;
    }

    /// Set parameters from Params struct
    pub fn set_params(&mut self, params: &Params) {
        self.params = SynthParams::from(params);
    }

    /// Check if the synth is currently producing sound
    pub fn is_playing(&self) -> bool {
        !self.voices.is_empty()
    }

    /// Get the current time
    pub fn current_time(&self) -> f32 {
        self.current_time
    }

    /// Reset the synth state
    pub fn reset(&mut self) {
        self.voices.clear();
        self.note_to_voice.clear();
        self.current_time = 0.0;
        self.pitch_bend = 0.0;
    }
}

/// Convert MIDI note number to frequency in Hz
pub fn midi_to_freq(midi_note: u8) -> f32 {
    440.0 * 2.0_f32.powf((midi_note as f32 - 69.0) / 12.0)
}

/// Simple soft clipping to prevent harsh distortion
fn soft_clip(x: f32) -> f32 {
    if x > 1.0 {
        1.0 - 1.0 / (1.0 + x)
    } else if x < -1.0 {
        -1.0 + 1.0 / (1.0 - x)
    } else {
        x
    }
}

/// Simple pseudo-random number generator (0.0 to 1.0)
fn rand_f32() -> f32 {
    // Use a simple LCG
    static mut SEED: u32 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED as f32) / (u32::MAX as f32)
    }
}

/// Render a composition to audio samples
pub fn render_to_samples(events: &[MidiEvent], sample_rate: u32) -> Vec<f32> {
    if events.is_empty() {
        return Vec::new();
    }

    // Find the end time of the last event plus some release time
    let end_time = events.iter().map(|e| e.time).fold(0.0f32, f32::max) + 1.0;
    let total_samples = (end_time * sample_rate as f32) as usize;

    let mut synth = BasicSynth::with_sample_rate(sample_rate);
    let mut samples = Vec::with_capacity(total_samples);
    let mut event_idx = 0;

    for _ in 0..total_samples {
        // Process any events at this time
        while event_idx < events.len() && events[event_idx].time <= synth.current_time() {
            synth.process_event(&events[event_idx]);
            event_idx += 1;
        }

        samples.push(synth.generate_sample());
    }

    samples
}

/// Render a composition to stereo audio samples
pub fn render_to_stereo_samples(events: &[MidiEvent], sample_rate: u32) -> Vec<f32> {
    let mono = render_to_samples(events, sample_rate);
    let mut stereo = Vec::with_capacity(mono.len() * 2);
    for sample in mono {
        stereo.push(sample);
        stereo.push(sample);
    }
    stereo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_midi_to_freq() {
        // A4 = 440 Hz
        assert!((midi_to_freq(69) - 440.0).abs() < 0.01);
        // C4 = ~261.63 Hz
        assert!((midi_to_freq(60) - 261.63).abs() < 0.1);
    }

    #[test]
    fn test_adsr_envelope() {
        let env = ADSREnvelope::default();

        // At start, amplitude should be 0
        assert!(env.amplitude(0.0, 0.0, None) < 0.01);

        // At peak of attack, should be ~1.0
        assert!((env.amplitude(env.attack_time, 0.0, None) - 1.0).abs() < 0.1);

        // After decay, should be at sustain level
        assert!((env.amplitude(env.attack_time + env.decay_time + 0.1, 0.0, None) - env.sustain_level).abs() < 0.1);
    }

    #[test]
    fn test_basic_synth() {
        let mut synth = BasicSynth::new();

        // Should start silent
        assert!(!synth.is_playing());

        // Note on
        synth.note_on(60, 0.8);
        assert!(synth.is_playing());

        // Generate some samples
        let samples = synth.generate_samples(1000);
        assert_eq!(samples.len(), 1000);

        // Should have non-zero samples
        assert!(samples.iter().any(|&s| s.abs() > 0.01));
    }
}
