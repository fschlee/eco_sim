use conrod_core as cc;
use conrod_core::{widget_ids, WidgetCommon};

widget_ids! {
    pub struct Ids {
        txt
    }
}

#[derive(WidgetCommon)]
pub struct MentalStateRepWidget {
    data: String,
    #[conrod(common_builder)]
    common: conrod_core::widget::CommonBuilder,
}

impl cc::Widget for MentalStateRepWidget {
    type State = Ids;
    type Style = ();
    type Event = ();

    fn init_state(&self, gen: cc::widget::id::Generator) -> Self::State {
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
        cc::widget::Text::new(&self.data)
            .font_size(super::theme().font_size_small)
            .parent(id)
            .set(state.txt, ui)
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
        arg.map(|msr| MentalStateRepWidget {
            common: Default::default(),
            data: msr.to_string(),
        })
        .collect()
    }
}
