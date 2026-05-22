use std::{f64::consts::PI, ops::Neg};

use derive_new::new;
use ordered_float::{Float, OrderedFloat, UniformOrdered};
use rand::{
    distributions::{
        uniform::{SampleUniform, UniformSampler},
        Distribution, Uniform,
    },
    Rng,
};
use rand_pcg::Pcg64;
use sdl2::{gfx::primitives::DrawRenderer, pixels::Color};
use serde::Serialize;

use crate::{
    core::{ActionReward, Env, EnvProperties},
    spaces::{BoxR, Discrete, Space},
    utils::{
        custom::{
            screen::{Screen, ScreenGuiTransformations},
            structs::Metadata,
            traits::Sample,
            types::O64,
            util_fns::clip,
        },
        renderer::{RenderMode, Renderer, Renders},
        seeding::{self, rand_random},
    },
};

/// An environment which implements the double pendulum (also known as the
/// acrobot) swing-up problem described in
/// [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
/// and the original [Acrobot paper](https://papers.nips.cc/paper/1995).
///
/// The system consists of two links connected linearly to form a chain, with
/// one end of the chain fixed. The joint between the two links is actuated. The
/// goal is to apply torques on the actuated joint to swing the free end of the
/// chain above a given height while starting from a downward hanging position.
///
/// The agent is rewarded `-1` for every step taken until the episode ends.
///
/// The episode ends when the following condition occurs:
///
/// 1. Termination: The free end of the second link reaches the target height,
///    i.e. `-cos(theta1) - cos(theta1 + theta2)` is greater than `1.0`.
#[derive(Debug, Clone, Serialize)]
pub struct DoublePendulumEnv {
    /// The available actions that can be taken.
    pub action_space: Discrete,
    /// The range of values that can be observed.
    pub observation_space: BoxR<DoublePendulumObservation>,
    /// The type of renders produced.
    pub render_mode: RenderMode,
    /// The current state of the environment.
    pub state: DoublePendulumObservation,
    /// Additional pieces of information provided by the environment.
    pub metadata: Metadata<Self>,
    /// The gravity constant applied to the environment.
    pub gravity: O64,
    /// The length of the first (fixed) link.
    pub link_length_1: O64,
    /// The length of the second (free) link.
    pub link_length_2: O64,
    /// The mass of the first link.
    pub link_mass_1: O64,
    /// The mass of the second link.
    pub link_mass_2: O64,
    /// The position of the center of mass of the first link.
    pub link_com_pos_1: O64,
    /// The position of the center of mass of the second link.
    pub link_com_pos_2: O64,
    /// The moment of inertia of both links.
    pub link_moi: O64,
    /// The maximum absolute angular velocity of the first link.
    pub max_vel_1: O64,
    /// The maximum absolute angular velocity of the second link.
    pub max_vel_2: O64,
    /// The magnitude of the torque applied to the actuated joint.
    pub torque: O64,
    /// The maximum absolute uniform noise added to the applied torque.
    pub torque_noise_max: O64,
    /// The number of seconds between state updates.
    pub dt: O64,
    renderer: Renderer,
    screen: Screen,
    #[serde(skip_serializing)]
    rand_random: Pcg64,
}

/// The torque applied to the actuated joint for each available action.
const AVAIL_TORQUE: [f64; 3] = [-1.0, 0.0, 1.0];

impl DoublePendulumEnv {
    /// Creates a double pendulum environment using the defaults from the paper.
    pub fn new(render_mode: RenderMode) -> Self {
        let (mut rand_random, _) = rand_random(None);

        let gravity = OrderedFloat(9.8);
        let link_length_1 = OrderedFloat(1.0);
        let link_length_2 = OrderedFloat(1.0);
        let link_mass_1 = OrderedFloat(1.0);
        let link_mass_2 = OrderedFloat(1.0);
        let link_com_pos_1 = OrderedFloat(0.5);
        let link_com_pos_2 = OrderedFloat(0.5);
        let link_moi = OrderedFloat(1.0);
        let max_vel_1 = OrderedFloat(4. * PI);
        let max_vel_2 = OrderedFloat(9. * PI);
        let torque = OrderedFloat(1.0);
        let torque_noise_max = OrderedFloat(0.0);
        let dt = OrderedFloat(0.2);

        let high = DoublePendulumObservation::new(
            OrderedFloat(PI),
            OrderedFloat(PI),
            max_vel_1,
            max_vel_2,
        );

        let action_space = Discrete(3);
        let observation_space = BoxR::new(-high, high);

        let renderer = Renderer::new(render_mode, None, None);

        let metadata = Metadata::default();
        let screen = Screen::new(
            500,
            500,
            "Double Pendulum",
            metadata.render_fps,
            render_mode,
        );

        let state = DoublePendulumObservation::sample_between(&mut rand_random, None);

        Self {
            action_space,
            observation_space,
            render_mode,
            state,
            metadata,
            gravity,
            link_length_1,
            link_length_2,
            link_mass_1,
            link_mass_2,
            link_com_pos_1,
            link_com_pos_2,
            link_moi,
            max_vel_1,
            max_vel_2,
            torque,
            torque_noise_max,
            dt,
            renderer,
            screen,
            rand_random,
        }
    }

    /// Computes the time derivative of the augmented state given an applied torque.
    ///
    /// The state passed in (and returned) is `[theta1, theta2, dtheta1, dtheta2]`.
    fn dsdt(&self, s: [f64; 4], torque: f64) -> [f64; 4] {
        let m1 = self.link_mass_1.into_inner();
        let m2 = self.link_mass_2.into_inner();
        let l1 = self.link_length_1.into_inner();
        let lc1 = self.link_com_pos_1.into_inner();
        let lc2 = self.link_com_pos_2.into_inner();
        let i1 = self.link_moi.into_inner();
        let i2 = self.link_moi.into_inner();
        let g = self.gravity.into_inner();

        let [theta1, theta2, dtheta1, dtheta2] = s;

        let d1 = m1 * lc1.powi(2)
            + m2 * (l1.powi(2) + lc2.powi(2) + 2. * l1 * lc2 * theta2.cos())
            + i1
            + i2;
        let d2 = m2 * (lc2.powi(2) + l1 * lc2 * theta2.cos()) + i2;
        let phi2 = m2 * lc2 * g * (theta1 + theta2 - PI / 2.).cos();
        let phi1 = -m2 * l1 * lc2 * dtheta2.powi(2) * theta2.sin()
            - 2. * m2 * l1 * lc2 * dtheta2 * dtheta1 * theta2.sin()
            + (m1 * lc1 + m2 * l1) * g * (theta1 - PI / 2.).cos()
            + phi2;
        let ddtheta2 =
            (torque + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1.powi(2) * theta2.sin() - phi2)
                / (m2 * lc2.powi(2) + i2 - d2.powi(2) / d1);
        let ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;

        [dtheta1, dtheta2, ddtheta1, ddtheta2]
    }

    /// Integrates the augmented state forward by `dt` using a fourth order Runge-Kutta scheme.
    fn rk4(&self, y0: [f64; 4], torque: f64, dt: f64) -> [f64; 4] {
        let add = |a: [f64; 4], b: [f64; 4], scale: f64| -> [f64; 4] {
            [
                a[0] + b[0] * scale,
                a[1] + b[1] * scale,
                a[2] + b[2] * scale,
                a[3] + b[3] * scale,
            ]
        };

        let k1 = self.dsdt(y0, torque);
        let k2 = self.dsdt(add(y0, k1, dt / 2.), torque);
        let k3 = self.dsdt(add(y0, k2, dt / 2.), torque);
        let k4 = self.dsdt(add(y0, k3, dt), torque);

        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = y0[i] + dt / 6. * (k1[i] + 2. * k2[i] + 2. * k3[i] + k4[i]);
        }
        out
    }

    fn render(
        mode: RenderMode,
        screen: &mut Screen,
        metadata: &Metadata<Self>,
        link_length_1: O64,
        link_length_2: O64,
        state: DoublePendulumObservation,
    ) -> Renders {
        assert!(metadata.render_modes.contains(&mode));

        screen.load_gui();
        screen.consume_events();

        let screen_dim = screen.screen_width() as f64;
        let l1 = link_length_1.into_inner();
        let l2 = link_length_2.into_inner();
        // Leave a 10% margin around the fully extended pendulum.
        let bound = (l1 + l2) * 1.1;
        let scale = screen_dim / (2. * bound);
        let offset = screen_dim / 2.;

        let theta1 = state.theta1.into_inner();
        let theta2 = state.theta2.into_inner();

        // Joint positions in a y-up coordinate frame; the screen applies a
        // vertical flip so the pendulum is displayed hanging downwards.
        let p0 = (offset, offset);
        let p1 = (
            offset + scale * l1 * theta1.sin(),
            offset - scale * l1 * theta1.cos(),
        );
        let p2 = (
            p1.0 + scale * l2 * (theta1 + theta2).sin(),
            p1.1 - scale * l2 * (theta1 + theta2).cos(),
        );

        screen.draw_on_canvas(
            |canvas| {
                canvas.set_draw_color(Color::WHITE);
                canvas.clear();

                // The target height the free end must reach to terminate.
                let goal_y = (offset + scale) as i16;
                canvas
                    .hline(0, screen_dim as i16, goal_y, Color::RGB(0, 180, 0))
                    .unwrap();

                let draw_link = |a: (f64, f64), b: (f64, f64), width: f64, color: Color| {
                    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
                    let len = (dx * dx + dy * dy).sqrt();
                    let (nx, ny) = if len > 0. {
                        (-dy / len * width / 2., dx / len * width / 2.)
                    } else {
                        (0., 0.)
                    };
                    let xs = [a.0 + nx, b.0 + nx, b.0 - nx, a.0 - nx].map(|v| v.floor() as i16);
                    let ys = [a.1 + ny, b.1 + ny, b.1 - ny, a.1 - ny].map(|v| v.floor() as i16);

                    canvas.aa_polygon(&xs, &ys, color).unwrap();
                    canvas.filled_polygon(&xs, &ys, color).unwrap();
                };

                draw_link(p0, p1, 8., Color::RGB(0, 102, 204));
                draw_link(p1, p2, 8., Color::RGB(204, 102, 0));

                for (x, y) in [p0, p1, p2] {
                    canvas
                        .filled_circle(
                            x.floor() as i16,
                            y.floor() as i16,
                            6,
                            Color::RGB(64, 64, 64),
                        )
                        .unwrap();
                    canvas
                        .aa_circle(x.floor() as i16, y.floor() as i16, 6, Color::BLACK)
                        .unwrap();
                }
            },
            ScreenGuiTransformations::default(),
        );

        screen.render(mode)
    }
}

const DOUBLE_PENDULUM_RENDER_MODES: &[RenderMode] = &[
    RenderMode::Human,
    RenderMode::RgbArray,
    RenderMode::SingleRgbArray,
    RenderMode::None,
];

impl Default for Metadata<DoublePendulumEnv> {
    fn default() -> Self {
        Metadata::new(DOUBLE_PENDULUM_RENDER_MODES, 15)
    }
}

/// Wraps `value` into the half-open interval `[low, high)`.
fn wrap(mut value: f64, low: f64, high: f64) -> f64 {
    let diff = high - low;
    while value >= high {
        value -= diff;
    }
    while value < low {
        value += diff;
    }
    value
}

/// Defines the state found in the double pendulum environment.
#[derive(new, Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct DoublePendulumObservation {
    /// The angle of the first link relative to the downward vertical.
    pub theta1: O64,
    /// The angle of the second link relative to the first link.
    pub theta2: O64,
    /// The angular velocity of the first link.
    pub theta1_dot: O64,
    /// The angular velocity of the second link.
    pub theta2_dot: O64,
}

impl From<DoublePendulumObservation> for Vec<f64> {
    fn from(observation: DoublePendulumObservation) -> Self {
        Vec::from_iter(
            [
                observation.theta1,
                observation.theta2,
                observation.theta1_dot,
                observation.theta2_dot,
            ]
            .iter()
            .map(|v| v.into_inner()),
        )
    }
}

impl Neg for DoublePendulumObservation {
    type Output = DoublePendulumObservation;

    fn neg(self) -> Self::Output {
        DoublePendulumObservation {
            theta1: -self.theta1,
            theta2: -self.theta2,
            theta1_dot: -self.theta1_dot,
            theta2_dot: -self.theta2_dot,
        }
    }
}

impl Sample for DoublePendulumObservation {
    fn sample_between<R: Rng>(rng: &mut R, bounds: Option<BoxR<Self>>) -> Self {
        let BoxR { low, high } = bounds.unwrap_or({
            let observation_bound = DoublePendulumObservation::new(
                OrderedFloat(0.1),
                OrderedFloat(0.1),
                OrderedFloat(0.1),
                OrderedFloat(0.1),
            );
            BoxR::new(-observation_bound, observation_bound)
        });

        Uniform::new(low, high).sample(rng)
    }
}

/// The sampler responsible for generating an observation using uniform probability.
pub struct UniformDoublePendulumObservation {
    theta1_sampler: UniformOrdered<f64>,
    theta2_sampler: UniformOrdered<f64>,
    theta1_dot_sampler: UniformOrdered<f64>,
    theta2_dot_sampler: UniformOrdered<f64>,
}

impl SampleUniform for DoublePendulumObservation {
    type Sampler = UniformDoublePendulumObservation;
}

impl UniformSampler for UniformDoublePendulumObservation {
    type X = DoublePendulumObservation;

    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: rand::distributions::uniform::SampleBorrow<Self::X> + Sized,
        B2: rand::distributions::uniform::SampleBorrow<Self::X> + Sized,
    {
        UniformDoublePendulumObservation {
            theta1_sampler: UniformOrdered::new(low.borrow().theta1, high.borrow().theta1),
            theta2_sampler: UniformOrdered::new(low.borrow().theta2, high.borrow().theta2),
            theta1_dot_sampler: UniformOrdered::new(
                low.borrow().theta1_dot,
                high.borrow().theta1_dot,
            ),
            theta2_dot_sampler: UniformOrdered::new(
                low.borrow().theta2_dot,
                high.borrow().theta2_dot,
            ),
        }
    }

    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: rand::distributions::uniform::SampleBorrow<Self::X> + Sized,
        B2: rand::distributions::uniform::SampleBorrow<Self::X> + Sized,
    {
        UniformDoublePendulumObservation {
            theta1_sampler: UniformOrdered::new_inclusive(
                low.borrow().theta1,
                high.borrow().theta1,
            ),
            theta2_sampler: UniformOrdered::new_inclusive(
                low.borrow().theta2,
                high.borrow().theta2,
            ),
            theta1_dot_sampler: UniformOrdered::new_inclusive(
                low.borrow().theta1_dot,
                high.borrow().theta1_dot,
            ),
            theta2_dot_sampler: UniformOrdered::new_inclusive(
                low.borrow().theta2_dot,
                high.borrow().theta2_dot,
            ),
        }
    }

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        DoublePendulumObservation {
            theta1: self.theta1_sampler.sample(rng),
            theta2: self.theta2_sampler.sample(rng),
            theta1_dot: self.theta1_dot_sampler.sample(rng),
            theta2_dot: self.theta2_dot_sampler.sample(rng),
        }
    }
}

impl Env for DoublePendulumEnv {
    type Action = usize;

    type Observation = DoublePendulumObservation;

    type Info = ();

    type ResetInfo = ();

    fn step(&mut self, action: Self::Action) -> ActionReward<Self::Observation, Self::Info> {
        assert!(
            self.action_space.contains(action),
            "{} usize invalid",
            action
        );

        let mut applied_torque = AVAIL_TORQUE[action] * self.torque.into_inner();

        let torque_noise_max = self.torque_noise_max.into_inner();
        if torque_noise_max > 0. {
            applied_torque +=
                Uniform::new(-torque_noise_max, torque_noise_max).sample(&mut self.rand_random);
        }

        let DoublePendulumObservation {
            theta1,
            theta2,
            theta1_dot,
            theta2_dot,
        } = self.state;

        let integrated = self.rk4(
            [
                theta1.into_inner(),
                theta2.into_inner(),
                theta1_dot.into_inner(),
                theta2_dot.into_inner(),
            ],
            applied_torque,
            self.dt.into_inner(),
        );

        let theta1 = OrderedFloat(wrap(integrated[0], -PI, PI));
        let theta2 = OrderedFloat(wrap(integrated[1], -PI, PI));
        let theta1_dot = clip(OrderedFloat(integrated[2]), -self.max_vel_1, self.max_vel_1);
        let theta2_dot = clip(OrderedFloat(integrated[3]), -self.max_vel_2, self.max_vel_2);

        self.state = DoublePendulumObservation {
            theta1,
            theta2,
            theta1_dot,
            theta2_dot,
        };

        let done = -theta1.cos() - (theta1 + theta2).cos() > OrderedFloat(1.0);
        let reward = if done {
            OrderedFloat(0.0)
        } else {
            OrderedFloat(-1.0)
        };

        let screen = &mut self.screen;
        let metadata = &self.metadata;
        let link_length_1 = self.link_length_1;
        let link_length_2 = self.link_length_2;
        let state = self.state;

        self.renderer.render_step(&mut |mode| {
            Self::render(mode, screen, metadata, link_length_1, link_length_2, state)
        });

        ActionReward {
            observation: self.state,
            reward,
            done,
            truncated: false,
            info: Some(()),
        }
    }

    fn reset(
        &mut self,
        seed: Option<u64>,
        return_info: bool,
        options: Option<BoxR<Self::Observation>>,
    ) -> (Self::Observation, Option<Self::ResetInfo>) {
        let (rand_random, _) = seeding::rand_random(seed);
        self.rand_random = rand_random;

        self.state = DoublePendulumObservation::sample_between(&mut self.rand_random, options);

        let screen = &mut self.screen;
        let metadata = &self.metadata;
        let link_length_1 = self.link_length_1;
        let link_length_2 = self.link_length_2;
        let state = self.state;

        self.renderer.reset();
        self.renderer.render_step(&mut |mode| {
            Self::render(mode, screen, metadata, link_length_1, link_length_2, state)
        });

        if return_info {
            (self.state, Some(()))
        } else {
            (self.state, None)
        }
    }

    fn render(&mut self, mode: RenderMode) -> Renders {
        let screen = &mut self.screen;
        let metadata = &self.metadata;
        let link_length_1 = self.link_length_1;
        let link_length_2 = self.link_length_2;
        let state = self.state;

        let render_fn =
            &mut |mode| Self::render(mode, screen, metadata, link_length_1, link_length_2, state);
        if self.render_mode != RenderMode::None {
            self.renderer.get_renders(render_fn)
        } else {
            render_fn(mode)
        }
    }

    fn close(&mut self) {
        self.screen.close();
    }
}

impl EnvProperties for DoublePendulumEnv {
    type ActionSpace = Discrete;

    type ObservationSpace = BoxR<DoublePendulumObservation>;

    fn metadata(&self) -> &Metadata<Self> {
        &self.metadata
    }

    fn rand_random(&self) -> &Pcg64 {
        &self.rand_random
    }

    fn action_space(&self) -> &Self::ActionSpace {
        &self.action_space
    }

    fn observation_space(&self) -> &Self::ObservationSpace {
        &self.observation_space
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_double_pendulum_when_stepped_then_state_remains_bounded() {
        let mut env = DoublePendulumEnv::new(RenderMode::None);
        env.reset(Some(0), false, None);

        for step in 0..2000 {
            let action = step % 3;
            let reward = env.step(action);
            let DoublePendulumObservation {
                theta1,
                theta2,
                theta1_dot,
                theta2_dot,
            } = reward.observation;

            assert!((-PI..=PI).contains(&theta1.into_inner()));
            assert!((-PI..=PI).contains(&theta2.into_inner()));
            assert!(theta1_dot.abs() <= env.max_vel_1);
            assert!(theta2_dot.abs() <= env.max_vel_2);
            assert!(reward.reward == OrderedFloat(-1.0) || reward.reward == OrderedFloat(0.0));

            if reward.done {
                env.reset(Some(0), false, None);
            }
        }
    }
}
