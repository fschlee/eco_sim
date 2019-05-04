
use std::time::Instant;
use conrod_core as cc;
use conrod_core::{widget_ids, widget, Colorable, Labelable, Positionable, Sizeable, Widget};

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
        hunger_dialer,
        number_dialer,
        plot_path,
        // Scrollbar
        canvas_scrollbar,

        //
        edit_canvas,
        action_text,
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

#[derive(Debug, Clone)]
pub enum Action {
    Pause,
    Unpause,
    Reset(bool),
    UpdateMentalState(eco_sim::MentalState),
    Hover(LogicalPosition),
    Move(eco_sim::Entity, LogicalPosition),
    HighlightVisibility(eco_sim::Entity),
    ClearHighlight,
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
    test: f32,
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
        Self{mouse_pos : LogicalPosition{x: 0.0, y:0.0}, hidpi_factor, size, conrod, ids, prev : Instant::now(), window, paused: true, edit_ent: None, test: 0.0 }
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
                            match self.edit_ent {
                                Some(ent) => actions.push(Action::HighlightVisibility(ent)),
                                None => actions.push(Action::ClearHighlight),
                            }
                            ui_updates.push(UIUpdate::ToolTip{ pos : self.mouse_pos, txt : "foo".to_string()});
                        }
                    WindowEvent::MouseInput {button : MouseButton::Left, state: ElementState::Pressed, .. } =>
                        {
                            if let Some(entity) = self.edit_ent  {
                                if game_state.is_within(self.mouse_pos) {
                                    actions.push(Action::Move(entity, self.mouse_pos));
                                }
                            }
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
        if eventfull || !self.paused {
            let ui = & mut self.conrod.set_widgets();
            cc::widget::Canvas::new().pad(0.0).scroll_kids_vertically().w_h(1280.0, 1024.0).set(self.ids.canvas, ui);
            if self.paused {
                cc::widget::Text::new("Paused").font_size(32).mid_top_of(self.ids.canvas).set(self.ids.title, ui);
            }
            if let Some(edit_ent) = self.edit_ent {
                if let Some(ms) = game_state.get_mental_state(&edit_ent) {
                    cc::widget::Canvas::new().pad(0.0)
                        .w_h(256.0, 1024.0)
                        .mid_right_of(self.ids.canvas)
                        .set(self.ids.edit_canvas, ui);
                    let txt = format!("{:?}", game_state.get_type(&edit_ent).unwrap());
                    cc::widget::Text::new(&txt).font_size(32).mid_top_of(self.ids.edit_canvas).set(self.ids.dialer_title, ui);

                    for hunger in cc::widget::number_dialer::NumberDialer::new(ms.hunger.0, 0.0, 10.0, 3)
                        .down_from(self.ids.dialer_title, 60.0)
                        .align_middle_x_of(self.ids.edit_canvas)
                        .w_h(160.0, 40.0)
                        .label("Hunger")
                        .set(self.ids.hunger_dialer, ui) {
                        let mut new_ms = ms.clone();
                        new_ms.hunger = eco_sim::Hunger(hunger);
                        actions.push(Action::UpdateMentalState(new_ms));
                    }
                    let act_text = match ms.current_action {

                        None => format!("Idle"),
                        Some(eco_sim::Action::Eat(food)) => format!("eating {:?}", game_state.get_type(&food).unwrap()),
                        Some(eco_sim::Action::Move(pos)) => format!("moving to {:?}", pos),
                    };
                    cc::widget::Text::new(&act_text).font_size(16)
                        .down_from(self.ids.hunger_dialer, 60.0)
                        .align_middle_x_of(self.ids.edit_canvas).set(self.ids.action_text, ui);
                    for sight in cc::widget::number_dialer::NumberDialer::new(ms.sight_radius as f32, 0.0, 20.0, 0)
                        .down_from(self.ids.action_text, 60.0)
                        .align_middle_x_of(self.ids.edit_canvas)
                        .w_h(160.0, 40.0)
                        .label("Sight")
                        .set(self.ids.number_dialer, ui) {
                        let mut new_ms = ms.clone();
                        new_ms.sight_radius = sight as u32;
                        actions.push(Action::UpdateMentalState(new_ms));
                    }

                }
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
                    actions.push(Action::Pause);
                }
            }
            _ => ()
        }
    }
}