//! Procedural macros for generating musical note constants
//!
//! Generates constants like `c4q` (C4 quarter note), `gs5h` (G#5 half note), etc.
//!
//! ## Accidentals
//! - `s` = sharp (cs = C#)
//! - `f` = flat (df = Db)
//! - `ss` = double sharp (css = C##)
//! - `ff` = double flat (dff = Dbb)
//! - `n` = natural (cn = C natural)
//! - `nn` = double natural (cnn = C double natural)
//!
//! ## Durations
//! - `w` = whole, `h` = half, `q` = quarter, `e` = eighth
//! - `i` = sixteenth, `t` = thirty-second, `x` = sixty-fourth, `o` = 128th
//! - Add `d` for dotted, `dd` for double-dotted, `ddd` for triple-dotted

use proc_macro::TokenStream;
use quote::quote;

/// All duration variants with their names and fractional values
const DURATIONS: &[(&str, &str, f32)] = &[
    // Base durations
    ("w", "Whole", 1.0),
    ("h", "Half", 0.5),
    ("q", "Quarter", 0.25),
    ("e", "Eighth", 0.125),
    ("i", "Sixteenth", 0.0625),
    ("t", "ThirtySecond", 0.03125),
    ("x", "SixtyFourth", 0.015625),
    ("o", "OneTwentyEighth", 0.0078125),
    // Single dotted (1.5x)
    ("wd", "DottedWhole", 1.5),
    ("hd", "DottedHalf", 0.75),
    ("qd", "DottedQuarter", 0.375),
    ("ed", "DottedEighth", 0.1875),
    ("id", "DottedSixteenth", 0.09375),
    ("td", "DottedThirtySecond", 0.046875),
    ("xd", "DottedSixtyFourth", 0.0234375),
    ("od", "DottedOneTwentyEighth", 0.01171875),
    // Double dotted (1.75x)
    ("wdd", "DoubleDottedWhole", 1.75),
    ("hdd", "DoubleDottedHalf", 0.875),
    ("qdd", "DoubleDottedQuarter", 0.4375),
    ("edd", "DoubleDottedEighth", 0.21875),
    ("idd", "DoubleDottedSixteenth", 0.109375),
    ("tdd", "DoubleDottedThirtySecond", 0.0546875),
    ("xdd", "DoubleDottedSixtyFourth", 0.02734375),
    ("odd", "DoubleDottedOneTwentyEighth", 0.013671875),
    // Triple dotted (1.875x)
    ("wddd", "TripleDottedWhole", 1.875),
    ("hddd", "TripleDottedHalf", 0.9375),
    ("qddd", "TripleDottedQuarter", 0.46875),
    ("eddd", "TripleDottedEighth", 0.234375),
    ("iddd", "TripleDottedSixteenth", 0.1171875),
    ("tddd", "TripleDottedThirtySecond", 0.05859375),
    ("xddd", "TripleDottedSixtyFourth", 0.029296875),
    ("oddd", "TripleDottedOneTwentyEighth", 0.0146484375),
];

/// All pitch variants with semitone offsets from C
const PITCHES: &[(&str, i32)] = &[
    // Natural notes
    ("c", 0), ("d", 2), ("e", 4), ("f", 5), ("g", 7), ("a", 9), ("b", 11),
    // Sharps
    ("cs", 1), ("ds", 3), ("es", 5), ("fs", 6), ("gs", 8), ("as", 10), ("bs", 12),
    // Flats
    ("cf", -1), ("df", 1), ("ef", 3), ("ff", 4), ("gf", 6), ("af", 8), ("bf", 10),
    // Double sharps
    ("css", 2), ("dss", 4), ("ess", 6), ("fss", 7), ("gss", 9), ("ass", 11), ("bss", 13),
    // Double flats
    ("cff", -2), ("dff", 0), ("eff", 2), ("fff", 3), ("gff", 5), ("aff", 7), ("bff", 9),
    // Naturals (same as base, but explicit)
    ("cn", 0), ("dn", 2), ("en", 4), ("fn", 5), ("gn", 7), ("an", 9), ("bn", 11),
    // Double naturals
    ("cnn", 0), ("dnn", 2), ("enn", 4), ("fnn", 5), ("gnn", 7), ("ann", 9), ("bnn", 11),
];

/// Generates all notes for octaves -1 to 9
#[proc_macro]
pub fn generate_all_notes(_input: TokenStream) -> TokenStream {
    let mut tokens = proc_macro2::TokenStream::new();

    // Octaves -1 through 9
    for octave in -1i32..=9 {
        for (pitch_name, semitone) in PITCHES {
            // Calculate MIDI note number: C4 = 60, C0 = 12, C-1 = 0
            let midi = ((octave + 1) * 12 + semitone).clamp(0, 127) as u8;

            // Octave name handling: -1 becomes "_1"
            let octave_str = if octave < 0 {
                format!("_{}", -octave)
            } else {
                octave.to_string()
            };

            for (dur_suffix, dur_variant, dur_value) in DURATIONS {
                let const_name = format!("{}{}{}", pitch_name, octave_str, dur_suffix);
                let const_ident = syn::Ident::new(&const_name, proc_macro2::Span::call_site());
                let dur_ident = syn::Ident::new(dur_variant, proc_macro2::Span::call_site());

                // Generate constant
                tokens.extend(quote! {
                    pub const #const_ident: Note = Note::Atom {
                        midi: #midi,
                        duration: Duration::#dur_ident,
                        velocity: 0.8,
                    };
                });

                // Generate macro for note with parameters (e.g., c4q!(pitch = env!(...)))
                tokens.extend(quote! {
                    #[macro_export]
                    macro_rules! #const_ident {
                        // No params - just return the note
                        () => {
                            $crate::Note::Atom {
                                midi: #midi,
                                duration: $crate::Duration::#dur_ident,
                                velocity: 0.8,
                            }
                        };
                        // With params - wrap in Serial with Param nodes
                        ($($key:ident = $value:expr),+ $(,)?) => {{
                            $crate::Note::Serial(vec![
                                $(
                                    $crate::Note::Param {
                                        key: stringify!($key).to_string(),
                                        value: $crate::param_value_from($value),
                                    },
                                )+
                                $crate::Note::Atom {
                                    midi: #midi,
                                    duration: $crate::Duration::#dur_ident,
                                    velocity: 0.8,
                                },
                            ])
                        }};
                    }
                });

                // Generate duration tie constants (just the duration, used to extend notes)
                // Only generate once per duration (not per pitch/octave)
                if pitch_name == &"c" && octave == 0 {
                    let tie_ident = syn::Ident::new(dur_suffix, proc_macro2::Span::call_site());
                    let dur_f32 = *dur_value;
                    tokens.extend(quote! {
                        pub const #tie_ident: Note = Note::DurationTie {
                            duration: #dur_f32,
                        };
                    });
                }
            }
        }
    }

    tokens.into()
}

/// Generates rest constants: rw, rh, rq, re, ri, rt, rx, ro and all dotted variants
#[proc_macro]
pub fn generate_rests(_input: TokenStream) -> TokenStream {
    let mut tokens = proc_macro2::TokenStream::new();

    for (dur_suffix, dur_variant, _) in DURATIONS {
        let const_name = format!("r{}", dur_suffix);
        let const_ident = syn::Ident::new(&const_name, proc_macro2::Span::call_site());
        let dur_ident = syn::Ident::new(dur_variant, proc_macro2::Span::call_site());

        tokens.extend(quote! {
            pub const #const_ident: Note = Note::Rest {
                duration: Duration::#dur_ident,
            };
        });
    }

    tokens.into()
}

/// Generates previous pitch constants: p, pw, ph, pq, pe, pi, pt, px, po and all dotted variants
#[proc_macro]
pub fn generate_previous_pitch(_input: TokenStream) -> TokenStream {
    let mut tokens = proc_macro2::TokenStream::new();

    // Previous pitch with no duration (uses previous duration)
    tokens.extend(quote! {
        pub const p: Note = Note::PreviousPitch {
            duration: None,
        };
    });

    // Previous pitch with explicit durations
    for (dur_suffix, dur_variant, _) in DURATIONS {
        let const_name = format!("p{}", dur_suffix);
        let const_ident = syn::Ident::new(&const_name, proc_macro2::Span::call_site());
        let dur_ident = syn::Ident::new(dur_variant, proc_macro2::Span::call_site());

        tokens.extend(quote! {
            pub const #const_ident: Note = Note::PreviousPitch {
                duration: Some(Duration::#dur_ident),
            };
        });
    }

    tokens.into()
}
