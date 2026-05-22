use gym_rs::{
    core::Env, envs::classical_control::double_pendulum::DoublePendulumEnv,
    utils::renderer::RenderMode,
};
use ordered_float::OrderedFloat;
use rand::{thread_rng, Rng};

fn main() {
    let mut env = DoublePendulumEnv::new(RenderMode::Human);
    env.reset(None, false, None);

    const N: usize = 15;
    let mut rewards = Vec::with_capacity(N);

    let mut rng = thread_rng();
    for _ in 0..N {
        let mut current_reward = OrderedFloat(0.);

        for _ in 0..500 {
            let action = rng.gen_range(0..=2);
            let state_reward = env.step(action);
            current_reward += state_reward.reward;

            if state_reward.done {
                break;
            }
        }

        env.reset(None, false, None);
        rewards.push(current_reward);
    }

    env.close();

    println!("{:?}", rewards);
}
