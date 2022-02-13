#![allow(unused_imports)]
use anyhow::Result;
use itertools::{multiunzip, multizip, zip};
use opencv::{
    calib3d::{find_fundamental_mat, FM_RANSAC},
    core::{no_array, KeyPoint, Point, Point2d, Point2f, Ptr, Scalar, NORM_HAMMING, NORM_L2},
    features2d::{draw_keypoints, BFMatcher, DrawMatchesFlags},
    highgui::{self, named_window, set_window_property, WINDOW_NORMAL, WND_PROP_TOPMOST},
    imgproc::{self, canny, cvt_color, good_features_to_track, COLOR_BGR2GRAY},
    prelude::{DescriptorMatcher, DescriptorMatcherConst, Feature2DTrait, Mat, ORB},
    types::{
        VectorOfDMatch, VectorOfKeyPoint, VectorOfPoint, VectorOfPoint2f, VectorOfVectorOfDMatch,
        VectorOfVectorOfPoint2f, VectorOfi32, VectorOfu8,
    },
    videostab::{VideoFileSource, VideoFileSourceTrait},
};
use sample_consensus::Estimator;

const WINDOW_NAME: &str = "gray";
const SELECT_GRAY_OUTPUT: i32 = 'g' as i32;
const SELECT_CANNY_OUTPUT: i32 = 'c' as i32;
const SELECT_ORIGINAL_OUTPUT: i32 = 'o' as i32;
const KEY_ESCAPE: i32 = 27;
const KEY_SPACE: i32 = 32;

fn main() -> Result<()> {
    println!("Running ...");

    let mut video = VideoFileSource::new("labeled/0.hevc", false)?;

    named_window(WINDOW_NAME, WINDOW_NORMAL)?;
    set_window_property(WINDOW_NAME, WND_PROP_TOPMOST, 1.0)?;
    let mut state = State::new()?;

    loop {
        state.input(video.next_frame()?);

        state.filter()?;

        state.extract()?;

        let output = state.visual_output()?;

        highgui::imshow(WINDOW_NAME, &output)?;
        let key_pressed = highgui::wait_key(50)?;
        match_key(&mut state, key_pressed)?;

        if state.halted {
            let key_pressed = highgui::wait_key(-1)?;
            match_key(&mut state, key_pressed)?;
        }

        state.forward_frame_state();
    }
}

fn match_key(state: &mut State, key_pressed: i32) -> Result<()> {
    match key_pressed {
        KEY_ESCAPE => std::process::exit(0),
        SELECT_GRAY_OUTPUT => state.select_visual_output_base(VisualOutputBase::Gray),
        SELECT_CANNY_OUTPUT => state.select_visual_output_base(VisualOutputBase::Canny),
        SELECT_ORIGINAL_OUTPUT => state.select_visual_output_base(VisualOutputBase::Original),
        KEY_SPACE => state.halted = true,
        _ => (),
    }

    Ok(())
}

struct State {
    previous_frame_state: Option<FrameState>,
    current_frame_state: Option<FrameState>,

    orb: Ptr<dyn ORB>,
    bf: Ptr<BFMatcher>,

    matches: Vec<Match>,

    visual_output_base: VisualOutputBase,
    halted: bool,
}

struct FrameState {
    original: Mat,
    gray: Mat,
    canny: Mat,

    features: VectorOfPoint,
    keypoints: VectorOfKeyPoint,
    descriptors: Mat,
}

impl State {
    fn new() -> Result<Self> {
        let orb = <dyn ORB>::default()?;
        let bf = BFMatcher::create(NORM_HAMMING, true)?;

        Ok(Self {
            previous_frame_state: None,
            current_frame_state: None,

            orb,
            bf,

            matches: vec![],

            visual_output_base: VisualOutputBase::Gray,
            halted: false,
        })
    }

    fn input(&mut self, input: Mat) {
        self.current_frame_state = Some(FrameState {
            original: input,
            gray: Mat::default(),
            canny: Mat::default(),

            features: VectorOfPoint::new(),
            keypoints: VectorOfKeyPoint::new(),
            descriptors: Mat::default(),
        })
    }

    fn filter(&mut self) -> Result<()> {
        let frame = self
            .current_frame_state
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("No previous frame."))?;

        cvt_color(&frame.original, &mut frame.gray, COLOR_BGR2GRAY, 0)?;
        canny(&frame.gray, &mut frame.canny, 60.0, 100.0, 3, false)?;
        Ok(())
    }

    fn extract(&mut self) -> Result<()> {
        let frame = self
            .current_frame_state
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("No previous frame."))?;

        good_features_to_track(
            &frame.gray,
            &mut frame.features,
            3000,
            0.01,
            3.0,
            &no_array(),
            3,
            false,
            0.004,
        )?;

        frame.keypoints = VectorOfKeyPoint::from_iter(frame.features.iter().map(|feature| {
            KeyPoint::new_point(
                Point2f::new(feature.x as f32, feature.y as f32),
                20.0,
                0.0,
                0.0,
                0,
                0,
            )
            .unwrap()
        }));

        frame.descriptors = Mat::default();

        self.orb
            .compute(&frame.gray, &mut frame.keypoints, &mut frame.descriptors)?;

        if let (Some(previous_frame), Some(current_frame)) =
            (&self.previous_frame_state, &self.current_frame_state)
        {
            let mut matches = VectorOfDMatch::new();
            self.bf.train_match(
                &current_frame.descriptors,
                &previous_frame.descriptors,
                &mut matches,
                // 1,
                &Mat::default(),
                // false,
            )?;

            let matches = matches
                .into_iter()
                // .filter_map(|matches| {
                //     let first = matches.get(0).ok()?;
                //     let second = matches.get(1).ok()?;
                //     if first.distance < 0.75 * second.distance {
                //         Some(first)
                //     } else {
                //         None
                //     }
                // })
                .map(|dmatch| {
                    let current_keypoint =
                        current_frame.keypoints.get(dmatch.query_idx as usize)?;
                    let previous_keypoint =
                        previous_frame.keypoints.get(dmatch.train_idx as usize)?;

                    Ok((current_keypoint, previous_keypoint))
                })
                .collect::<Result<Vec<_>>>()?;
            let (current_keypoints, previous_keypoints): (Vec<KeyPoint>, Vec<KeyPoint>) =
                multiunzip(matches);

            let mut mask = VectorOfu8::with_capacity(current_keypoints.len());

            let _ = find_fundamental_mat(
                &VectorOfPoint2f::from_iter(current_keypoints.iter().map(|p| p.pt)),
                &VectorOfPoint2f::from_iter(previous_keypoints.iter().map(|p| p.pt)),
                FM_RANSAC,
                5.0,
                0.1,
                100,
                &mut mask,
            )?;

            self.matches = multizip((current_keypoints, previous_keypoints, mask))
                .filter_map(|(current_keypoint, previous_keypoint, filter)| {
                    if filter > 0 {
                        Some(Match {
                            current_keypoint,
                            previous_keypoint,
                        })
                    } else {
                        None
                    }
                })
                .collect();
        } else {
            self.matches = vec![];
        }

        Ok(())
    }

    fn visual_output(&self) -> Result<Mat> {
        let frame = self
            .current_frame_state
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No previous frame."))?;

        let mut output = Mat::default();

        let selected_output = match self.visual_output_base {
            VisualOutputBase::Gray => &frame.gray,
            VisualOutputBase::Canny => &frame.canny,
            VisualOutputBase::Original => &frame.original,
        };

        draw_keypoints(
            &selected_output,
            &frame.keypoints,
            &mut output,
            Scalar::new(0.0, 0.0, 255.0, 255.0),
            DrawMatchesFlags::DEFAULT,
        )?;

        // cvt_color(&input, &mut gray, COLOR_BGR2GRAY, 0)?;

        for m in &self.matches {
            imgproc::line(
                &mut output,
                Point {
                    x: m.current_keypoint.pt.x as i32,
                    y: m.current_keypoint.pt.y as i32,
                },
                Point {
                    x: m.previous_keypoint.pt.x as i32,
                    y: m.previous_keypoint.pt.y as i32,
                },
                Scalar::new(0.0, 255.0, 0.0, 255.0),
                1,
                imgproc::LINE_8,
                0,
            )?;
        }

        Ok(output)
    }

    fn forward_frame_state(&mut self) {
        self.previous_frame_state = self.current_frame_state.take();
    }

    fn select_visual_output_base(&mut self, visual_output_base: VisualOutputBase) {
        self.visual_output_base = visual_output_base;
    }
}

#[derive(Debug)]
enum VisualOutputBase {
    Gray,
    Canny,
    Original,
}

struct Match {
    current_keypoint: KeyPoint,
    previous_keypoint: KeyPoint,
}
