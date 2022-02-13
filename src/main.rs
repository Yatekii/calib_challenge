#![allow(unused_imports)]
use anyhow::Result;
use opencv::{
    core::{no_array, Ptr, Scalar},
    features2d::{draw_keypoints, DrawMatchesFlags},
    highgui::{self, named_window, set_window_property, WINDOW_NORMAL, WND_PROP_TOPMOST},
    imgproc::{self, canny, cvt_color, good_features_to_track, COLOR_BGR2GRAY},
    prelude::{Feature2DTrait, Mat, ORB},
    types::{VectorOfKeyPoint, VectorOfPoint},
    videostab::{VideoFileSource, VideoFileSourceTrait},
};

const WINDOW_NAME: &str = "gray";
const SELECT_GRAY_OUTPUT: i32 = 'g' as i32;
const SELECT_CANNY_OUTPUT: i32 = 'c' as i32;
const SELECT_ORIGINAL_OUTPUT: i32 = 'o' as i32;

fn main() -> Result<()> {
    println!("Running ...");

    let mut video = VideoFileSource::new("labeled/0.hevc", false)?;

    named_window(WINDOW_NAME, WINDOW_NORMAL)?;
    set_window_property(WINDOW_NAME, WND_PROP_TOPMOST, 1.0)?;
    let mut state = State::new()?;

    loop {
        state.input(video.next_frame()?);

        // for row in 0..gray.rows() {
        //     for col in 0..gray.cols() {
        //         let elem = gray.at_2d_mut::<u8>(row, col)?;
        //         *elem = elem.saturating_mul(2);
        //     }
        // }

        state.filter()?;

        state.extract()?;

        let output = state.visual_output()?;

        highgui::imshow(WINDOW_NAME, &output)?;
        let key_pressed = highgui::wait_key(50)?;
        match key_pressed {
            27 => return Ok(()),
            SELECT_GRAY_OUTPUT => state.select_visual_output_base(VisualOutputBase::Gray),
            SELECT_CANNY_OUTPUT => state.select_visual_output_base(VisualOutputBase::Canny),
            SELECT_ORIGINAL_OUTPUT => state.select_visual_output_base(VisualOutputBase::Original),
            _ => (),
        }

        state.forward_frame_state();
    }
}

struct State {
    previous_frame_state: Option<FrameState>,
    current_frame_state: Option<FrameState>,

    orb: Ptr<dyn ORB>,

    visual_output_base: VisualOutputBase,
}

struct FrameState {
    original: Mat,
    gray: Mat,
    canny: Mat,

    features: VectorOfPoint,
    keypoints: VectorOfKeyPoint,
}

impl State {
    fn new() -> Result<Self> {
        let orb = <dyn ORB>::default()?;
        Ok(Self {
            previous_frame_state: None,
            current_frame_state: None,

            orb,

            visual_output_base: VisualOutputBase::Gray,
        })
    }

    fn input(&mut self, input: Mat) {
        self.current_frame_state = Some(FrameState {
            original: input,
            gray: Mat::default(),
            canny: Mat::default(),

            features: VectorOfPoint::new(),
            keypoints: VectorOfKeyPoint::new(),
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

        let mask = Mat::default();
        self.orb.detect(&frame.canny, &mut frame.keypoints, &mask)?;

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

        for feature in &frame.features {
            imgproc::circle(
                &mut output,
                feature,
                10,
                Scalar::new(0.0, 255.0, 0.0, 255.0),
                2,
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
