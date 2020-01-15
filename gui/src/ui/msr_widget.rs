use crate::ui::sphere_packing::SpherePacking;
use conrod_core as cc;
use conrod_core::position::Position::Relative;
use conrod_core::{widget_ids, Colorable, Positionable, Sizeable, Ui, WidgetCommon};
use itertools::Itertools;

widget_ids! {
    pub struct Ids {
        name,
        txt[],
        back[],
        bars[],
        action_canvas[],
        action_txt[],
        behavior_txt[],
        behavior_canvas[],
    }
}

#[derive(WidgetCommon)]
pub struct MentalStateRepWidget {
    data: String,
    entity: eco_sim::WorldEntity,
    bars: Vec<(String, Vec<f32>)>,
    actions: Vec<(eco_sim::Action, Vec<usize>)>,
    behaviors: Vec<(Option<eco_sim::Behavior>, Vec<usize>)>,
    #[conrod(common_builder)]
    common: conrod_core::widget::CommonBuilder,
}

impl cc::Widget for MentalStateRepWidget {
    type State = Ids;
    type Style = ();
    type Event = ();

    fn init_state(&self, mut gen: cc::widget::id::Generator) -> Self::State {
        Ids::new(gen)
    }

    fn style(&self) -> Self::Style {
        ()
    }

    fn update(self, args: cc::widget::UpdateArgs<Self>) -> Self::Event {
        let cc::widget::UpdateArgs {
            id,
            state,
            style,
            rect,
            ui,
            ..
        } = args;
        let back_c = self.bars.len();
        let bar_c = self.bars.get(0).map_or(0, |t| t.1.len());
        state.update(|ids| {
            ids.txt.resize(back_c + 2, &mut ui.widget_id_generator());
            ids.back.resize(back_c, &mut ui.widget_id_generator());
            ids.bars
                .resize(back_c * bar_c, &mut ui.widget_id_generator());
            ids.action_txt.resize(bar_c, &mut ui.widget_id_generator());
            ids.behavior_txt
                .resize(bar_c, &mut ui.widget_id_generator());
            ids.action_canvas
                .resize(self.actions.len(), &mut ui.widget_id_generator());
            ids.behavior_canvas
                .resize(self.behaviors.len(), &mut ui.widget_id_generator());
        });

        let colors = distinct_colors(bar_c);
        debug_assert!(colors.len() >= bar_c);
        let mut bar_ids = state.bars.iter();
        let w = 120.0;
        let h = 20.0;
        let name = self.entity.to_string();
        cc::widget::Text::new(&name)
            .font_size(ui.theme().font_size_medium)
            .parent(id)
            .mid_top_of(id)
            .set(state.name, ui);
        let mut last = state.name;
        for ((back_id, txt_id), (txt, bars)) in state
            .txt
            .iter()
            .dropping(2)
            .zip(state.back.iter())
            .zip(self.bars)
        {
            cc::widget::BorderedRectangle::new(cc::Dimensions::from([w, h]))
                .parent(id)
                .down_from(last, 20.0)
                .set(*back_id, ui);
            last = *back_id;

            for ((bar, b_id), [r, g, b]) in bars.iter().zip(bar_ids.by_ref()).zip(colors.iter()) {
                cc::widget::Rectangle::fill_with(
                    cc::Dimensions::from([10.0, 20.0]),
                    cc::Color::Rgba(*r, *g, *b, 0.8),
                )
                .parent(*back_id)
                .x_position_relative_to(
                    *back_id,
                    cc::position::Relative::Scalar(w * (*bar) as f64 - 0.5 * w),
                )
                .set(*b_id, ui);
            }
            cc::widget::Text::new(&txt)
                .font_size(ui.theme().font_size_small)
                .parent(*back_id)
                .middle_of(*back_id)
                .set(*txt_id, ui);
        }
        let mut action_ids = state.action_txt.iter();
        for ((action, indices), action_canvas_id) in
            self.actions.iter().zip(state.action_canvas.iter())
        {
            let txt = action.to_string();
            let mut prev = None;
            cc::widget::Canvas::new()
                .w_h(120.0, 20.0)
                .align_left_of(id)
                .down(5.0)
                .set(*action_canvas_id, ui);
            for ((s, idx), a_id) in substrings(indices.len(), &txt)
                .zip(indices)
                .zip(action_ids.by_ref())
            {
                let [r, g, b] = colors[*idx];
                let mut w = cc::widget::Text::new(s)
                    .font_size(ui.theme().font_size_small)
                    .align_left_of(*action_canvas_id)
                    .color(cc::Color::Rgba(r, g, b, 1.0));
                //.set(*a_id, ui);
                if let Some(prev_id) = prev {
                    w.right_from(prev_id, 0.0).set(*a_id, ui);
                } else {
                    w.set(*a_id, ui);
                }
                prev = Some(*a_id);
            }
        }
        let mut behavior_ids = state.behavior_txt.iter();
        for ((behavior, indices), behavior_canvas_id) in
            self.behaviors.iter().zip(state.behavior_canvas.iter())
        {
            let txt = eco_sim::Behavior::fmt(behavior);
            let mut prev = None;
            cc::widget::Canvas::new()
                .w_h(120.0, 20.0)
                .align_left_of(id)
                .down(5.0)
                .set(*behavior_canvas_id, ui);
            for ((s, idx), a_id) in substrings(indices.len(), &txt)
                .zip(indices)
                .zip(behavior_ids.by_ref())
            {
                let [r, g, b] = colors[*idx];
                let mut w = cc::widget::Text::new(s)
                    .font_size(ui.theme().font_size_small)
                    .align_left_of(*behavior_canvas_id)
                    .color(cc::Color::Rgba(r, g, b, 1.0));
                //.set(*a_id, ui);
                if let Some(prev_id) = prev {
                    w.right_from(prev_id, 0.0).set(*a_id, ui);
                } else {
                    w.set(*a_id, ui);
                }
                prev = Some(*a_id);
            }
        }
    }
    fn default_y_dimension(&self, ui: &Ui) -> cc::position::Dimension {
        let bk = self.bars.len() as f64;
        let ac = self.actions.len() as f64;
        let bh = self.behaviors.len() as f64;

        cc::position::Dimension::Absolute(bk * 40.0 + ac * 60.0 + bh * 60.0)
    }
}

impl crate::simulation::MentalModelFn for MentalStateRepWidget {
    type Output = Vec<MentalStateRepWidget>;

    fn call<
        'a,
        MSR: eco_sim::agent::estimator::MentalStateRep + 'a,
        I: Iterator<Item = &'a MSR>,
    >(
        arg: I,
    ) -> Self::Output {
        arg.map(|msr| {
            let entity = msr.into_ms().id;
            let mut hunger = Vec::new();
            let mut tiredness = Vec::new();
            let mut aggression = Vec::new();
            let mut fear = Vec::new();
            let mut actions = Vec::new();
            let mut behaviors = Vec::new();
            let mut prefs: Vec<_> = eco_sim::EntityType::iter()
                .filter(|et| entity.e_type().can_eat(et))
                .map(|et| (et, Vec::new()))
                .collect();
            msr.iter().enumerate().for_each(|(idx, (e, act, beh))| {
                hunger.push(e.hunger().0);
                tiredness.push(e.tiredness().0);
                aggression.push(e.aggression().0);
                fear.push(e.fear().0);
                for (et, v) in prefs.iter_mut() {
                    v.push(e.pref(*et).0)
                }
                if let Some((_, indices)) = actions
                    .iter_mut()
                    .find(|(a, _): &&mut (eco_sim::Action, Vec<usize>)| *a == *act)
                {
                    indices.push(idx);
                } else {
                    actions.push((*act, vec![idx]));
                }

                if let Some((_, indices)) = behaviors
                    .iter_mut()
                    .find(|(b, _): &&mut (Option<eco_sim::Behavior>, Vec<usize>)| *b == *beh)
                {
                    indices.push(idx);
                } else {
                    behaviors.push((beh.clone(), vec![idx]));
                }
            });
            let mut bars = vec![
                ("Hunger".to_owned(), hunger),
                ("Tiredness".to_owned(), tiredness),
                ("Aggression".to_owned(), aggression),
                ("Fear".to_owned(), fear),
            ];
            bars.extend(prefs.into_iter().map(|(t, v)| (t.to_string(), v)));
            MentalStateRepWidget {
                entity,
                bars,
                actions,
                behaviors,
                common: Default::default(),
                data: msr.to_string(),
            }
        })
        .collect()
    }
}
type RGB = [f32; 3];

const W_R: f64 = 0.299;
const W_G: f64 = 1.0 - W_R - W_B;
const W_B: f64 = 0.114;
const U_MAX: f64 = 0.436;
const V_MAX: f64 = 0.615;

fn yuv_to_rgb(y: f64, u: f64, v: f64) -> RGB {
    const V_R: f64 = (1.0 - W_R) / V_MAX;
    const U_G: f64 = W_B * (1.0 - W_B) / (U_MAX * W_G);
    const V_G: f64 = W_R * (1.0 - W_R) / (V_MAX * W_G);
    const U_B: f64 = (1.0 - W_B) / U_MAX;
    let r = y + v * V_R;
    let g = y - u * U_G - v * V_G;
    let b = y + u * U_B;
    [clip(r), clip(g), clip(b)]
}
fn distinct_colors(n: usize) -> Vec<RGB> {
    SpherePacking::fit_n_in_box(n as u64, [0.2, 0.0, 0.0], [0.8, 1.0, 1.0])
        .take(n)
        .map(|[y, u, v]| yuv_to_rgb(y, u - U_MAX, v - V_MAX))
        .collect()
}

fn clip(c: f64) -> f32 {
    (c as f32).min(1.0).max(0.0)
}

fn substrings<'a>(n: usize, full_str: &'a str) -> impl Iterator<Item = &'a str> + 'a {
    let stride = if n <= full_str.len() {
        full_str.len() / n
    } else {
        1
    };
    (0..full_str.len())
        .step_by(stride)
        .enumerate()
        .map(move |(i, s)| {
            if s + stride >= full_str.len() || i == n - 1 {
                &full_str[s..]
            } else {
                &full_str[s..s + stride]
            }
        })
}
