
use std::time::Instant;
use conrod_core as cc;

use conrod_core::{widget_ids, widget::Widget, position::Positionable};

use crate::simulation::GameState;
use winit::{EventsLoop, Event, WindowEvent, MouseButton, ElementState, KeyboardInput, VirtualKeyCode, ModifiersState, dpi::{LogicalPosition, LogicalSize} };

widget_ids! {
    pub struct WidgetIds {
        // The scrollable canvas.
        canvas,
        // The title and introduction widgets.
        title,
        introduction,
        // Shapes.
        shapes_canvas,
        rounded_rectangle,
        shapes_left_col,
        shapes_right_col,
        shapes_title,
        line,
        point_path,
        rectangle_fill,
        rectangle_outline,
        trapezoid,
        oval_fill,
        oval_outline,
        circle,
        // Image.
        image_title,
        rust_logo,
        // Button, XyPad, Toggle.
        button_title,
        button,
        xy_pad,
        toggle,
        ball,
        // NumberDialer, PlotPath
        dialer_title,
        number_dialer,
        plot_path,
        // Scrollbar
        canvas_scrollbar,
    }
}
pub fn theme() -> conrod_core::Theme {
    use conrod_core::position::{Align, Direction, Padding, Position, Relative};
    conrod_core::Theme {
        name: "Demo Theme".to_string(),
        padding: Padding::none(),
        x_position: Position::Relative(Relative::Align(Align::Start), None),
        y_position: Position::Relative(Relative::Direction(Direction::Backwards, 20.0), None),
        background_color: conrod_core::color::TRANSPARENT,
        shape_color: conrod_core::color::LIGHT_CHARCOAL,
        border_color: conrod_core::color::BLACK,
        border_width: 0.0,
        label_color: conrod_core::color::WHITE,
        font_id: None,
        font_size_large: 26,
        font_size_medium: 18,
        font_size_small: 12,
        widget_styling: conrod_core::theme::StyleMap::default(),
        mouse_drag_threshold: 0.0,
        double_click_threshold: std::time::Duration::from_millis(500),
    }
}

pub enum UIUpdate {
    ToolTip{ pos : LogicalPosition, txt : String},
    MoveCamera { transformation: nalgebra::Translation3<f32> },
    Resized { size : LogicalSize},
    Refresh,
}

static CAMERA_STEP : f32 = 0.05;

pub struct UIState<'a> {
    window: &'a winit::Window,
    mouse_pos : LogicalPosition,
    edit_ent : Option<eco_sim::Entity>,
    hidpi_factor: f64,
    pub conrod: cc::Ui,
    pub ids: WidgetIds,
    size: LogicalSize,
    prev: Instant,
    paused: bool,
}
#[derive(Debug, Clone, Copy)]
pub enum Action {
    Pause,
    Unpause,
    Reset(bool),
    UpdateMentalState(eco_sim::MentalState),
    Hover(LogicalPosition),
}
impl<'a> UIState<'a> {
    pub fn new<'b: 'a>(window: &'b winit::Window) -> UIState<'a>{
        let size = window.get_inner_size().expect("window should have size");
        let hidpi_factor = window.get_hidpi_factor();
        let dim = [size.width, size.height];
        let mut conrod = cc::UiBuilder::new(dim).theme(theme()).build();
        let bytes: &[u8] = include_bytes!("../resources/fonts/NotoSans/NotoSans-Regular.ttf");
        let font = cc::text::Font::from_bytes(bytes).unwrap();
        conrod.fonts.insert(font);
        let ids = WidgetIds::new(conrod.widget_id_generator());
        {
            let mut ui = conrod.set_widgets();
            widgets(&mut ui, &ids);
        }
        Self{mouse_pos : LogicalPosition{x: 0.0, y:0.0}, hidpi_factor, size, conrod, ids, prev : Instant::now(), window, paused: true, edit_ent: None }
    }
    pub fn process(&mut self, event_loop: &mut EventsLoop, game_state : &GameState) -> (bool, Vec<UIUpdate>, Vec<Action>) {
        let mut should_close = false;
        let mut eventfull = false;
        let mut actions = vec![];
        let mut ui_updates = vec![];
        event_loop.poll_events(|event| {
            if let Some(event) = conrod_winit::convert_event(event.clone(), self.window) {
                self.conrod.handle_event(event);
            }
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => should_close = true,
                    WindowEvent::Resized(logical_size) => ui_updates.push(UIUpdate::Resized { size : logical_size}),
                    WindowEvent::CursorMoved { position, .. } => {
                        self.mouse_pos = position;
                        actions.push(Action::Hover(position));
                    },
                    WindowEvent::MouseInput {button : MouseButton::Right, state: ElementState::Pressed, .. } =>
                        {
                            self.edit_ent = game_state.get_editable_entity(self.mouse_pos);
                            ui_updates.push(UIUpdate::ToolTip{ pos : self.mouse_pos, txt : "foo".to_string()});
                        }
                    WindowEvent::KeyboardInput { input : KeyboardInput{ virtual_keycode : Some(key), modifiers, state: ElementState::Released, .. }, .. } =>
                        {
                            self.process_key(&mut ui_updates, &mut actions, key, modifiers);
                        }
                    _ => ()

                },
                _ => (),
            }
            ()
        }
        );
        if self.conrod.global_input().events().next().is_some() {
            eventfull = true;
        }
        let now = Instant::now();
        let delta = now - self.prev;
        if eventfull {
            let ui = & mut self.conrod.set_widgets();
            cc::widget::Canvas::new().pad(0.0).scroll_kids_vertically().set(self.ids.canvas, ui);
            if self.paused {
                cc::widget::Text::new("Paused").font_size(32).mid_top_of(self.ids.canvas).set(self.ids.title, ui);
            }






        }
        (should_close, ui_updates, actions)
    }
    fn process_key(&mut self, ui_updates: &mut Vec<UIUpdate>, actions: &mut Vec<Action>, key : VirtualKeyCode, modifiers : ModifiersState ) {
        use VirtualKeyCode::*;
        match key {
            VirtualKeyCode::Up => ui_updates.push(UIUpdate::MoveCamera { transformation : nalgebra::Translation3::new(0.0, -CAMERA_STEP, 0.0)}),
            VirtualKeyCode::Down => ui_updates.push(UIUpdate::MoveCamera { transformation : nalgebra::Translation3::new(0.0, CAMERA_STEP, 0.0)}),
            VirtualKeyCode::Left => ui_updates.push(UIUpdate::MoveCamera { transformation : nalgebra::Translation3::new(-CAMERA_STEP, 0.0, 0.0)}),
            VirtualKeyCode::Right => ui_updates.push(UIUpdate::MoveCamera { transformation : nalgebra::Translation3::new(CAMERA_STEP, 0.0, 0.0)}),
            F5 => actions.push(Action::Reset(self.paused)),
            F12 => ui_updates.push(UIUpdate::Refresh),
            VirtualKeyCode::Pause | VirtualKeyCode::Space => {
                if self.paused {
                    self.paused = false;
                    actions.push(Action::Unpause);
                }
                else {
                    self.paused = true;
                    actions.push(Action::Unpause);
                }
            }
            _ => ()
        }
    }
}
fn draw_map(ui: & mut cc::UiCell, ids: &WidgetIds, draw_data:  impl Iterator<Item = (usize, usize, eco_sim::ViewData)>){
    use conrod_core::{widget, Widget};
    const MARGIN: conrod_core::Scalar = 30.0;
    const SHAPE_GAP: conrod_core::Scalar = 50.0;
    const TITLE_SIZE: conrod_core::FontSize = 42;
    const SUBTITLE_SIZE: conrod_core::FontSize = 32;

    // `Canvas` is a widget that provides some basic functionality for laying out children widgets.
    // By default, its size is the size of the window. We'll use this as a background for the
    // following widgets, as well as a scrollable container for the children widgets.
    const TITLE: &'static str = "All Widgets";
    widget::Canvas::new().pad(MARGIN).scroll_kids_vertically().set(ids.canvas, ui);


}

fn widgets(ui: & mut cc::UiCell, ids: &WidgetIds) {
    use conrod_core::{widget, Colorable, Labelable, Positionable, Sizeable, Widget};
    use std::iter::once;

    const MARGIN: conrod_core::Scalar = 30.0;
    const SHAPE_GAP: conrod_core::Scalar = 50.0;
    const TITLE_SIZE: conrod_core::FontSize = 42;
    const SUBTITLE_SIZE: conrod_core::FontSize = 32;

    // `Canvas` is a widget that provides some basic functionality for laying out children widgets.
    // By default, its size is the size of the window. We'll use this as a background for the
    // following widgets, as well as a scrollable container for the children widgets.
    const TITLE: &'static str = "All Widgets";
    widget::Canvas::new().pad(MARGIN).scroll_kids_vertically().set(ids.canvas, ui);


    ////////////////
    ///// TEXT /////
    ////////////////


    // We'll demonstrate the `Text` primitive widget by using it to draw a title and an
    // introduction to the example.
    widget::Text::new(TITLE).font_size(TITLE_SIZE).mid_top_of(ids.canvas).set(ids.title, ui);

    const INTRODUCTION: &'static str =
        "This example aims to demonstrate all widgets that are provided by conrod.\
        \n\nThe widget that you are currently looking at is the Text widget. The Text widget \
        is one of several special \"primitive\" widget types which are used to construct \
        all other widget types. These types are \"special\" in the sense that conrod knows \
        how to render them via `conrod_core::render::Primitive`s.\
        \n\nScroll down to see more widgets!";
    widget::Text::new(INTRODUCTION)
        .padded_w_of(ids.canvas, MARGIN)
        .down(60.0)
        .align_middle_x_of(ids.canvas)
        .center_justify()
        .line_spacing(5.0)
        .set(ids.introduction, ui);


    ////////////////////////////
    ///// Lines and Shapes /////
    ////////////////////////////


    widget::Text::new("Lines and Shapes")
        .down(70.0)
        .align_middle_x_of(ids.canvas)
        .font_size(SUBTITLE_SIZE)
        .set(ids.shapes_title, ui);

    // Lay out the shapes in two horizontal columns.
    //
    // TODO: Have conrod provide an auto-flowing, fluid-list widget that is more adaptive for these
    // sorts of situations.
    widget::Canvas::new()
        .down(0.0)
        .align_middle_x_of(ids.canvas)
        .kid_area_w_of(ids.canvas)
        .h(360.0)
        .color(conrod_core::color::TRANSPARENT)
        .pad(MARGIN)
        .flow_down(&[
            (ids.shapes_left_col, widget::Canvas::new()),
            (ids.shapes_right_col, widget::Canvas::new()),
        ])
        .set(ids.shapes_canvas, ui);

    let shapes_canvas_rect = ui.rect_of(ids.shapes_canvas).unwrap();
    let w = shapes_canvas_rect.w();
    let h = shapes_canvas_rect.h() * 5.0 / 6.0;
    let radius = 10.0;
    widget::RoundedRectangle::fill([w, h], radius)
        .color(conrod_core::color::CHARCOAL.alpha(0.25))
        .middle_of(ids.shapes_canvas)
        .set(ids.rounded_rectangle, ui);

    let start = [-40.0, -40.0];
    let end = [40.0, 40.0];
    widget::Line::centred(start, end).mid_left_of(ids.shapes_left_col).set(ids.line, ui);

    let left = [-40.0, -40.0];
    let top = [0.0, 40.0];
    let right = [40.0, -40.0];
    let points = once(left).chain(once(top)).chain(once(right));
    widget::PointPath::centred(points).right(SHAPE_GAP).set(ids.point_path, ui);

    widget::Rectangle::fill([80.0, 80.0]).right(SHAPE_GAP).set(ids.rectangle_fill, ui);

    widget::Rectangle::outline([80.0, 80.0]).right(SHAPE_GAP).set(ids.rectangle_outline, ui);

    let bl = [-40.0, -40.0];
    let tl = [-20.0, 40.0];
    let tr = [20.0, 40.0];
    let br = [40.0, -40.0];
    let points = once(bl).chain(once(tl)).chain(once(tr)).chain(once(br));
    widget::Polygon::centred_fill(points).mid_left_of(ids.shapes_right_col).set(ids.trapezoid, ui);

    widget::Oval::fill([40.0, 80.0]).right(SHAPE_GAP + 20.0).align_middle_y().set(ids.oval_fill, ui);

    widget::Oval::outline([80.0, 40.0]).right(SHAPE_GAP + 20.0).align_middle_y().set(ids.oval_outline, ui);

    widget::Circle::fill(40.0).right(SHAPE_GAP).align_middle_y().set(ids.circle, ui);


    /////////////////
    ///// Image /////
    /////////////////


    widget::Text::new("Image")
        .down_from(ids.shapes_canvas, MARGIN)
        .align_middle_x_of(ids.canvas)
        .font_size(SUBTITLE_SIZE)
        .set(ids.image_title, ui);

    const LOGO_SIDE: conrod_core::Scalar = 144.0;


    /////////////////////////////////
    ///// Button, XYPad, Toggle /////
    /////////////////////////////////


    widget::Text::new("Button, XYPad and Toggle")
        .down_from(ids.rust_logo, 60.0)
        .align_middle_x_of(ids.canvas)
        .font_size(SUBTITLE_SIZE)
        .set(ids.button_title, ui);
    widget::Text::new("NumberDialer and PlotPath")
        .down_from(ids.image_title,  MARGIN)
        .align_middle_x_of(ids.canvas)
        .font_size(SUBTITLE_SIZE)
        .set(ids.dialer_title, ui);

    // Use a `NumberDialer` widget to adjust the frequency of the sine wave below.
    let min = 0.5;
    let max = 200.0;
    let mut sine_frequency = 12.0;
    let decimal_precision = 1;
    for new_freq in widget::NumberDialer::new(sine_frequency, min, max, decimal_precision)
        .down(60.0)
        .align_middle_x_of(ids.canvas)
        .w_h(160.0, 40.0)
        .label("F R E Q")
        .set(ids.number_dialer, ui)
        {
            sine_frequency = new_freq;
        }

    // Use the `PlotPath` widget to display a sine wave.
    let min_x = 0.0;
    let max_x = std::f32::consts::PI * 2.0 * sine_frequency;
    let min_y = -1.0;
    let max_y = 1.0;
    widget::PlotPath::new(min_x, max_x, min_y, max_y, f32::sin)
        .kid_area_w_of(ids.canvas)
        .h(240.0)
        .down(60.0)
        .align_middle_x_of(ids.canvas)
        .set(ids.plot_path, ui);


    /////////////////////
    ///// Scrollbar /////
    /////////////////////


    widget::Scrollbar::y_axis(ids.canvas).auto_hide(true).set(ids.canvas_scrollbar, ui);
}