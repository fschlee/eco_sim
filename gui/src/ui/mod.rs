mod conrod_winit_helper;
mod msr_widget;

use conrod_core as cc;
use conrod_core::{
    widget, widget_ids, Borderable, Colorable, Labelable, Positionable, Sizeable, UiCell, Widget,
};
use conrod_winit::{convert_event, WinitWindow};
use std::time::Instant;

use crate::simulation::GameState;
use crate::BackendSelection;
use conrod_core::widget::Id;
use eco_sim::entity_type::{Count, EntityType};
use eco_sim::WorldEntity;
use log::info;
use winit::{
    dpi::{LogicalPosition, LogicalSize, PhysicalSize},
    event::{
        ElementState, Event, KeyboardInput, ModifiersState, MouseButton, VirtualKeyCode,
        WindowEvent,
    },
    event_loop::EventLoop,
    window::Window,
};

pub type Queue<T> = std::collections::VecDeque<T>;

widget_ids! {
    pub struct WidgetIds {
        canvas,
        title,
        dialer_title,
        hunger_dialer,
        aggression_dialer,
        fear_dialer,
        tiredness_dialer,
        number_dialer,
        score_text,
        plot_path,
        canvas_scrollbar,
        edit_canvas,
        action_text,
        behavior_text,
        food_prefs[],
        list_canvas,
        food_pref_text,
        mental_model_canvas,
        mental_models[],
        mm_title,
        tooltip,
        tooltip_head,
        tooltip_text,

        menu_back_canvas,
        menu_canvas,
        menu_title,
        adapter_label,
        backend_label,
        backends,
        adapters,
        select_button,

    }
}

const HOVER_TIME: f32 = 0.5;

pub fn entity_type_label(et: EntityType) -> &'static str {
    use EntityType::*;
    match et {
        Rock => "Rock",
        Tree => "Tree",
        Grass => "Grass",
        Clover => "Clover",
        Rabbit => "Rabbit",
        Deer => "Deer",
        Wolf => "Wolf",
        Burrow => "Burrow",
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
    MoveCamera {
        transformation: nalgebra::Translation3<f32>,
    },
    Resized {
        size: PhysicalSize,
    },
    Refresh,
}

#[derive(Debug, Clone, Copy)]
pub enum AppState {
    Continue,
    Exit,
    ChangeAdapter(BackendSelection, usize),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum UiStatus {
    Running,
    Paused,
    Menu,
}

pub enum Action {
    Pause,
    Unpause,
    Reset(bool),
    UpdateMentalState(
        eco_sim::WorldEntity,
        Box<dyn FnMut(&mut eco_sim::MentalState)>,
    ),
    Hover(LogicalPosition),
    Move(eco_sim::WorldEntity, LogicalPosition),
    HighlightVisibility(eco_sim::WorldEntity),
    ClearHighlight,
    ToggleThreatMode,
    ToggleMapKnowledgeMode,
}
static CAMERA_STEP: f32 = 0.05;

pub type AdapterList = Vec<(BackendSelection, Vec<(usize, String)>)>;

pub struct UIState {
    // general
    pub window: Window,
    event_happened: bool,
    size: LogicalSize,
    pub conrod: cc::Ui,
    pub ids: WidgetIds,
    pub actions: Queue<Action>,
    pub ui_updates: Queue<UIUpdate>,
    pub app_state: AppState,
    ui_status: UiStatus,

    // for sim
    last_draw: Instant,
    mouse_pos: LogicalPosition,
    hover_pos: Option<eco_sim::Position>,
    hover_start_time: Instant,
    tooltip_index: usize,
    tooltip_active: bool,
    receiving_text: bool,
    paused: bool,
    edit_ent: Option<eco_sim::WorldEntity>,
    mental_model: Option<eco_sim::WorldEntity>,

    // for menu
    adapter_list: AdapterList,
    backends: Vec<String>,
    backend_selected: Option<usize>,
    adapters: Vec<String>,
    adapter_selected: Option<usize>,
}

impl UIState {
    pub fn new(window: Window, adapter_list: AdapterList) -> UIState {
        let size = window.inner_size();
        let dim = [size.width, size.height];
        let mut conrod = cc::UiBuilder::new(dim).theme(theme()).build();
        let bytes: &[u8] = include_bytes!("../../resources/fonts/NotoSans/NotoSans-Regular.ttf");
        let font = cc::text::Font::from_bytes(bytes).unwrap();
        conrod.fonts.insert(font);
        let mut ids = WidgetIds::new(conrod.widget_id_generator());
        ids.food_prefs.resize(
            EntityType::iter().count(),
            &mut conrod.widget_id_generator(),
        );
        let max_adapters = adapter_list.iter().map(|(b, v)| v.len()).max().unwrap_or(0);
        ids.mental_models
            .resize(10, &mut conrod.widget_id_generator());
        Self {
            mouse_pos: LogicalPosition { x: 0.0, y: 0.0 },
            size,
            conrod,
            ids,
            window,
            paused: true,
            edit_ent: None,
            mental_model: None,
            hover_pos: None,
            hover_start_time: Instant::now(),
            tooltip_index: 0,
            tooltip_active: false,
            receiving_text: false,
            event_happened: false,
            last_draw: Instant::now(),
            actions: Queue::new(),
            ui_updates: Queue::new(),
            app_state: AppState::Continue,
            ui_status: UiStatus::Running,

            adapters: Vec::new(),
            backends: adapter_list.iter().map(|(b, v)| b.to_string()).collect(),
            adapter_list,
            backend_selected: None,
            adapter_selected: None,
        }
    }
    pub fn process(&mut self, event: Event<()>, game_state: &GameState) -> AppState {
        if let Some(event) = conrod_winit_helper::convert_event(event.clone(), self) {
            self.conrod.handle_event(event);
        }
        let open = self.ui_status != UiStatus::Menu;
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => self.app_state = AppState::Exit,
                WindowEvent::Resized(logical_size) => {
                    self.ui_updates.push_back(UIUpdate::Resized {
                        size: logical_size.to_physical(self.window.hidpi_factor()),
                    });
                    self.size = self.window.inner_size();
                }
                WindowEvent::CursorMoved { position, .. } if open => {
                    self.mouse_pos = position;
                    let sim_pos = game_state.logical_to_sim_position(position);
                    if Some(sim_pos) != self.hover_pos {
                        self.hover_pos = Some(sim_pos);
                        self.hover_start_time = Instant::now();
                        self.tooltip_active = false;
                        self.tooltip_index = game_state.get_editable_index(sim_pos);
                    }
                    self.actions.push_back(Action::Hover(position));
                }
                WindowEvent::MouseInput {
                    button: MouseButton::Left,
                    state: ElementState::Pressed,
                    modifiers,
                    ..
                } if open && game_state.is_within(self.mouse_pos) => {
                    let entity = match self.hover_pos {
                        Some(hover_pos) if self.tooltip_active => {
                            game_state.get_nth_entity(self.tooltip_index, hover_pos)
                        }
                        _ => game_state.get_editable_entity(self.mouse_pos),
                    };
                    if modifiers.ctrl {
                        self.mental_model = entity;
                    } else {
                        self.edit_ent = entity;
                        match self.edit_ent {
                            Some(ent) => self.actions.push_back(Action::HighlightVisibility(ent)),
                            None => self.actions.push_back(Action::ClearHighlight),
                        }
                    }
                }
                WindowEvent::MouseInput {
                    button: MouseButton::Right,
                    state: ElementState::Pressed,
                    ..
                } if open && game_state.is_within(self.mouse_pos) => {
                    if let Some(entity) = self.edit_ent {
                        self.actions.push_back(Action::Move(entity, self.mouse_pos));
                    }
                }
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(key),
                            modifiers,
                            state: ElementState::Released,
                            ..
                        },
                    ..
                } => match self.ui_status {
                    UiStatus::Running | UiStatus::Paused => self.process_key(key, modifiers),
                    UiStatus::Menu => {
                        if let VirtualKeyCode::Escape = key {
                            self.resume_from_menu()
                        }
                    }
                },
                _ => (),
            },
            _ => (),
        }

        if self.conrod.global_input().events().next().is_some() {
            self.event_happened = true;
        }
        self.app_state
    }
    pub fn update(&mut self, game_state: &GameState) {
        match self.ui_status {
            UiStatus::Running => self.redraw(game_state),
            UiStatus::Paused if self.event_happened => self.redraw(game_state),
            UiStatus::Paused => (),
            UiStatus::Menu => {
                let mut resume = false;
                {
                    let ui = &mut self.conrod.set_widgets();
                    cc::widget::Canvas::new()
                        .pad(10.0)
                        .w_h(self.size.width, self.size.height)
                        .color(cc::Color::Rgba(0.5, 0.5, 0.5, 0.5))
                        .set(self.ids.menu_back_canvas, ui);
                    cc::widget::Canvas::new()
                        .pad(10.0)
                        .w_h(440.0, 300.0)
                        .mid_top_of(self.ids.menu_back_canvas)
                        .color(cc::color::BLUE)
                        .set(self.ids.menu_canvas, ui);
                    cc::widget::Text::new("Menu")
                        .mid_top_of(self.ids.menu_canvas)
                        .set(self.ids.menu_title, ui);
                    for idx in cc::widget::DropDownList::new(&self.backends, self.backend_selected)
                        .w_h(120.0, 120.0)
                        .max_visible_height(260.0)
                        .label("Graphic API")
                        .top_left_of(self.ids.menu_canvas)
                        .down(40.0)
                        .set(self.ids.backends, ui)
                    {
                        if Some(idx) != self.backend_selected {
                            self.backend_selected = Some(idx);
                            let (_, v) = &self.adapter_list[idx];
                            self.adapters = v.iter().map(|(i, s)| s.clone()).collect();
                        }
                    }
                    for idx in cc::widget::DropDownList::new(&self.adapters, self.adapter_selected)
                        .w_h(200.0, 120.0)
                        .max_visible_height(260.0)
                        .label("Graphic Adapter")
                        .right_from(self.ids.backends, 10.0)
                        .set(self.ids.adapters, ui)
                    {
                        if Some(idx) != self.adapter_selected {
                            self.adapter_selected = Some(idx);
                        }
                    }
                    for btn in cc::widget::Button::new()
                        .parent(self.ids.menu_canvas)
                        .right_from(self.ids.adapters, 10.0)
                        .w_h(80.0, 40.0)
                        .label("Select")
                        .set(self.ids.select_button, ui)
                    {
                        if let (Some(b_idx), Some(a_idx)) =
                            (self.backend_selected, self.adapter_selected)
                        {
                            if let Some((back, idx, name)) = self
                                .adapter_list
                                .get(b_idx)
                                .and_then(|(b, v)| v.get(a_idx).map(|(i, s)| (b, i, s)))
                            {
                                info!("should choose {:?} : {} ({})", back, name, idx);
                                self.app_state = AppState::ChangeAdapter(*back, *idx);
                                resume = true;
                            }
                        }
                    }
                }

                if resume {
                    self.resume_from_menu()
                }
            }
        }
    }
    fn redraw(&mut self, game_state: &GameState) {
        self.event_happened = false;
        let mut extend = 0;
        {
            let now = Instant::now();
            if now.duration_since(self.hover_start_time).as_secs_f32() > HOVER_TIME {
                self.tooltip_active = true;
            }
        }
        {
            const DOWN: f64 = 30.0;
            let mouse_pos = self.logical_to_conrod(self.mouse_pos);
            let ui = &mut self.conrod.set_widgets();

            cc::widget::Canvas::new()
                .pad(0.0)
                .scroll_kids_vertically()
                .w_h(self.size.width, self.size.height)
                .set(self.ids.canvas, ui);
            if self.paused {
                cc::widget::Text::new("Paused")
                    .font_size(32)
                    .mid_top_of(self.ids.canvas)
                    .set(self.ids.title, ui);
            }
            cc::widget::Canvas::new()
                .pad(0.0)
                .parent(self.ids.canvas)
                .w_h(256.0, self.size.height)
                .mid_right_of(self.ids.canvas)
                //                 .left_from(self.ids.edit_canvas, 60.0)
                .set(self.ids.mental_model_canvas, ui);
            if let Some(mm) = self.mental_model {
                let title = format!("{}", mm);
                cc::widget::Text::new(&title)
                    .font_size(32)
                    .mid_top_of(self.ids.mental_model_canvas)
                    .set(self.ids.mm_title, ui);
                let mut prev = self.ids.mm_title;
                let mut i = 0;
                if let Some(mental_model) =
                    game_state.with_mental_model::<msr_widget::MentalStateRepWidget>(&mm)
                {
                    for item in mental_model {
                        if i >= self.ids.mental_models.len() {
                            extend += 1;
                            continue;
                        }
                        let m_id = self.ids.mental_models[i];
                        i += 1;
                        item.parent(self.ids.mental_model_canvas)
                            .down_from(prev, 60.0)
                            .align_left()
                            .set(m_id, ui);
                        prev = m_id;
                    }
                }
            }
            if let Some(edit_ent) = self.edit_ent {
                cc::widget::Canvas::new()
                    .pad(0.0)
                    .parent(self.ids.canvas)
                    .w_h(256.0, 1024.0)
                    //        .mid_right_of(self.ids.canvas)
                    .left_from(self.ids.mental_model_canvas, 60.0)
                    // .right(1024.0)
                    .set(self.ids.edit_canvas, ui);
                let txt = format!("{}", edit_ent);
                cc::widget::Text::new(&txt)
                    .font_size(32)
                    .mid_top_of(self.ids.edit_canvas)
                    .set(self.ids.dialer_title, ui);
                if let Some(ms) = game_state.get_mental_state(&edit_ent) {
                    const MIN: f32 = 0.0;
                    const MAX: f32 = 1.0;
                    const P: u8 = 3;
                    const W: f64 = 170.0;
                    const H: f64 = 40.0;
                    let score = format!("Score: {}", ms.score);
                    cc::widget::text::Text::new(&score)
                        .font_size(16)
                        .down_from(self.ids.dialer_title, DOWN)
                        .align_middle_x_of(self.ids.edit_canvas)
                        .set(self.ids.score_text, ui);
                    for tiredness in cc::widget::number_dialer::NumberDialer::new(
                        ms.emotional_state.tiredness().0,
                        MIN,
                        MAX,
                        P,
                    )
                    .down_from(self.ids.score_text, DOWN)
                    .align_middle_x_of(self.ids.edit_canvas)
                    .w_h(W, H)
                    .label("Tiredness")
                    .set(self.ids.tiredness_dialer, ui)
                    {
                        self.actions.push_back(Action::UpdateMentalState(
                            ms.id,
                            Box::new(move |ms| {
                                ms.emotional_state
                                    .set_tiredness(eco_sim::Tiredness(tiredness))
                            }),
                        ));
                    }
                    for fear in cc::widget::number_dialer::NumberDialer::new(
                        ms.emotional_state.fear().0,
                        MIN,
                        MAX,
                        P,
                    )
                    .down_from(self.ids.tiredness_dialer, DOWN)
                    .align_middle_x_of(self.ids.edit_canvas)
                    .w_h(W, H)
                    .label("Fear")
                    .set(self.ids.fear_dialer, ui)
                    {
                        self.actions.push_back(Action::UpdateMentalState(
                            ms.id,
                            Box::new(move |ms| ms.emotional_state.set_fear(eco_sim::Fear(fear))),
                        ));
                    }
                    for aggro in cc::widget::number_dialer::NumberDialer::new(
                        ms.emotional_state.aggression().0,
                        MIN,
                        MAX,
                        P,
                    )
                    .down_from(self.ids.fear_dialer, DOWN)
                    .align_middle_x_of(self.ids.edit_canvas)
                    .w_h(W, H)
                    .label("Aggression")
                    .set(self.ids.aggression_dialer, ui)
                    {
                        self.actions.push_back(Action::UpdateMentalState(
                            ms.id,
                            Box::new(move |ms| {
                                ms.emotional_state
                                    .set_aggression(eco_sim::Aggression(aggro))
                            }),
                        ));
                    }
                    for hunger in cc::widget::number_dialer::NumberDialer::new(
                        ms.emotional_state.hunger().0,
                        MIN,
                        MAX,
                        P,
                    )
                    .down_from(self.ids.aggression_dialer, DOWN)
                    .align_middle_x_of(self.ids.edit_canvas)
                    .w_h(W, H)
                    .label("Hunger")
                    .set(self.ids.hunger_dialer, ui)
                    {
                        self.actions.push_back(Action::UpdateMentalState(
                            ms.id,
                            Box::new(move |ms| {
                                ms.emotional_state.set_hunger(eco_sim::Hunger(hunger))
                            }),
                        ));
                    }
                    let act_text = format!("{}", ms.current_action);
                    cc::widget::Text::new(&act_text)
                        .font_size(16)
                        .down_from(self.ids.hunger_dialer, DOWN)
                        .align_middle_x_of(self.ids.edit_canvas)
                        .set(self.ids.action_text, ui);
                    let beh_text = eco_sim::Behavior::fmt(&ms.current_behavior);
                    cc::widget::Text::new(&beh_text)
                        .font_size(16)
                        .down_from(self.ids.action_text, DOWN)
                        .align_middle_x_of(self.ids.edit_canvas)
                        .set(self.ids.behavior_text, ui);
                    for sight in cc::widget::number_dialer::NumberDialer::new(
                        ms.sight_radius as f32,
                        0.0,
                        20.0,
                        0,
                    )
                    .down_from(self.ids.behavior_text, DOWN)
                    .align_middle_x_of(self.ids.edit_canvas)
                    .w_h(160.0, 40.0)
                    .label("Sight")
                    .set(self.ids.number_dialer, ui)
                    {
                        self.actions.push_back(Action::UpdateMentalState(
                            ms.id,
                            Box::new(move |ms| ms.sight_radius = sight as eco_sim::Coord),
                        ));
                    }
                    cc::widget::Canvas::new()
                        .pad(0.0)
                        .w_h(256.0, (H + DOWN) * EntityType::COUNT as f64)
                        .parent(self.ids.edit_canvas)
                        .mid_right_of(self.ids.edit_canvas)
                        .down_from(self.ids.number_dialer, 2.0 * DOWN)
                        .set(self.ids.list_canvas, ui);
                    cc::widget::Text::new("Food preferences")
                        .font_size(20)
                        .mid_top_of(self.ids.list_canvas)
                        .set(self.ids.food_pref_text, ui);

                    let mut prev = self.ids.food_pref_text;
                    for (i, (&id, (et, old_pref))) in self
                        .ids
                        .food_prefs
                        .iter()
                        .zip(ms.food_preferences())
                        .enumerate()
                    {
                        let w = cc::widget::number_dialer::NumberDialer::new(old_pref, MIN, MAX, P)
                            .down_from(prev, DOWN)
                            .align_middle_x_of(self.ids.list_canvas)
                            .w_h(W, H)
                            .label(entity_type_label(et));

                        prev = id;

                        for fp in w.set(id, ui) {
                            self.actions.push_back(Action::UpdateMentalState(
                                ms.id,
                                Box::new(move |ms| ms.emotional_state.set_preference(et, fp)),
                            ));
                        }
                    }
                }
            }

            match self.hover_pos {
                Some(pos) if self.tooltip_active => {
                    if let Some(tt) = game_state.get_nth_entity(self.tooltip_index, pos) {
                        if let Some(ps) = game_state.get_physical_state(&tt) {
                            let (w, h) = (130.0, 120.0);
                            cc::widget::Canvas::new()
                                .pad(1.0)
                                .w_h(w, h)
                                .x_y(mouse_pos.0 + 0.5 * w, mouse_pos.1 - 0.5 * h)
                                .rgb(0.0, 0.0, 0.0)
                                .border_color(cc::color::WHITE)
                                .parent(self.ids.canvas)
                                .set(self.ids.tooltip, ui);
                            let head = format!("{}", tt);
                            cc::widget::Text::new(&head)
                                .font_size(16)
                                .mid_top_of(self.ids.tooltip)
                                .set(self.ids.tooltip_head, ui);
                            let txt = format!("{}", ps);
                            cc::widget::Text::new(&txt)
                                .font_size(12)
                                .align_middle_x_of(self.ids.tooltip)
                                .down_from(self.ids.tooltip_head, 20.0)
                                .set(self.ids.tooltip_text, ui);
                        } else {
                            let (w, h) = (90.0, 30.0);
                            cc::widget::Canvas::new()
                                .pad(1.0)
                                .w_h(w, h)
                                .x_y(mouse_pos.0 + 0.5 * w, mouse_pos.1 - 0.5 * h)
                                .rgb(0.0, 0.0, 0.0)
                                .border_color(cc::color::WHITE)
                                .parent(self.ids.canvas)
                                .set(self.ids.tooltip, ui);
                            let head = format!("{}", tt);
                            cc::widget::Text::new(&head)
                                .font_size(16)
                                .mid_top_of(self.ids.tooltip)
                                .set(self.ids.tooltip_head, ui);
                        }
                    } else if pos.within_bounds() {
                        let (w, h) = (90.0, 30.0);
                        cc::widget::Canvas::new()
                            .pad(1.0)
                            .w_h(w, h)
                            .x_y(mouse_pos.0 + 0.5 * w, mouse_pos.1 - 0.5 * h)
                            .rgb(0.0, 0.0, 0.0)
                            .border_color(cc::color::WHITE)
                            .parent(self.ids.canvas)
                            .set(self.ids.tooltip, ui);
                        let head = format!("{}", pos);
                        cc::widget::Text::new(&head)
                            .font_size(12)
                            .mid_top_of(self.ids.tooltip)
                            .set(self.ids.tooltip_head, ui);
                    }
                }
                _ => (),
            }
        }
        if extend > 0 {
            self.ids.mental_models.resize(
                self.ids.mental_models.len() + extend,
                &mut self.conrod.widget_id_generator(),
            );
        }
    }
    fn process_key(&mut self, key: VirtualKeyCode, modifiers: ModifiersState) {
        use VirtualKeyCode::*;
        match key {
            Space | M | T if self.receiving_text => (),
            Up => self.ui_updates.push_back(UIUpdate::MoveCamera {
                transformation: nalgebra::Translation3::new(0.0, -CAMERA_STEP, 0.0),
            }),
            Down => self.ui_updates.push_back(UIUpdate::MoveCamera {
                transformation: nalgebra::Translation3::new(0.0, CAMERA_STEP, 0.0),
            }),
            Left => self.ui_updates.push_back(UIUpdate::MoveCamera {
                transformation: nalgebra::Translation3::new(-CAMERA_STEP, 0.0, 0.0),
            }),
            Right => self.ui_updates.push_back(UIUpdate::MoveCamera {
                transformation: nalgebra::Translation3::new(CAMERA_STEP, 0.0, 0.0),
            }),
            F5 => self.actions.push_back(Action::Reset(self.paused)),
            F12 => self.ui_updates.push_back(UIUpdate::Refresh),
            Pause | Space => {
                if self.paused {
                    self.paused = false;
                    self.actions.push_back(Action::Unpause);
                } else {
                    self.paused = true;
                    self.actions.push_back(Action::Pause);
                }
            }
            Tab => {
                self.tooltip_index += 1;
            }
            T => self.actions.push_back(Action::ToggleThreatMode),
            M => self.actions.push_back(Action::ToggleMapKnowledgeMode),
            Escape => match self.ui_status {
                UiStatus::Running | UiStatus::Paused => self.ui_status = UiStatus::Menu,
                UiStatus::Menu => self.ui_status = UiStatus::Running,
            },
            _ => (),
        }
    }
    fn logical_to_conrod(&self, position: LogicalPosition) -> (cc::Scalar, cc::Scalar) {
        let (width, height) = self.get_inner_size().expect("cannot access windows size");
        (
            position.x - (width as cc::Scalar / 2.0),
            -position.y + (height as cc::Scalar / 2.0),
        )
    }
    fn resume_from_menu(&mut self) {
        self.event_happened = true;
        self.ui_status = if self.paused {
            UiStatus::Paused
        } else {
            UiStatus::Running
        }
    }
}

impl WinitWindow for UIState {
    fn get_inner_size(&self) -> Option<(u32, u32)> {
        Some(self.window.inner_size().into())
    }

    fn hidpi_factor(&self) -> f32 {
        self.window.hidpi_factor() as f32
    }
}
