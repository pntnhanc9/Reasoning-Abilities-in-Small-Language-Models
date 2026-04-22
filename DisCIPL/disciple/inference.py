import asyncio
import copy
from datetime import datetime

import numpy as np
from llamppl.inference.smc_record import SMCRecord
from llamppl.util import logsumexp


async def smc_standard(
    model,
    n_particles: int,
    ess_threshold: float = 0.5,
    max_steps: int = 1000,
    visualize: bool = False,
):
    """
    Standard sequential Monte Carlo algorithm with multinomial resampling.

    Modified version of llamppl.inference.smc_standard

    Args:
        model (llamppl.modeling.Model): The model to perform inference on.
        n_particles (int): Number of particles to execute concurrently.
        ess_threshold (float): Effective sample size below which resampling is triggered, given as a fraction of `n_particles`.
        max_steps (int): Maximum number of steps to run the algorithm.

    Returns:
        particles (list[llamppl.modeling.Model]): The completed particles after inference.
    """
    if visualize:
        visualization_dir = "html"
        json_file = "results/output.json"

    particles = [copy.deepcopy(model) for _ in range(n_particles)]
    await asyncio.gather(*[p.start() for p in particles])
    history = SMCRecord(n_particles) if visualize else None

    ancestor_indices = list(range(n_particles))
    did_resample = False

    step = 0
    while any(map(lambda p: not p.done_stepping(), particles)):

        # Step each particle
        for p in particles:
            p.untwist()
        await asyncio.gather(*[p.step() for p in particles if not p.done_stepping()])

        # Record history
        if visualize:
            if len(history.history) == 0:
                history.add_init(particles)
            elif did_resample:
                history.add_resample(ancestor_indices, particles)
            else:
                history.add_smc_step(particles)

            for p in particles:
                if p.proposal_context is not None:
                    print("Proposal Context:")
                    print(
                        p.proposal_context.lm.tokenizer.decode(
                            p.proposal_context.tokens
                        )
                    )
                    print()

                print("Completion:")
                print(p.context)
                print()

        # Avoid resampling on the last step
        if not all(map(lambda p: p.done_stepping(), particles)):

            # Normalize weights
            W = np.array([p.weight for p in particles])
            w_sum = logsumexp(W)
            normalized_weights = W - w_sum

            # Resample if necessary
            if -logsumexp(normalized_weights * 2) < np.log(ess_threshold) + np.log(
                n_particles
            ):
                # Alternative implementation uses a multinomial distribution and only makes n-1 copies, reusing existing one, but fine for now
                probs = np.exp(normalized_weights)
                ancestor_indices = [
                    np.random.choice(range(len(particles)), p=probs)
                    for _ in range(n_particles)
                ]

                if visualize:
                    # Sort the ancestor indices
                    ancestor_indices.sort()

                particles = [copy.deepcopy(particles[i]) for i in ancestor_indices]
                avg_weight = w_sum - np.log(n_particles)
                for p in particles:
                    p.weight = avg_weight

                did_resample = True
            else:
                did_resample = False

        step += 1
        if step > max_steps:
            break

    if hasattr(model, "check"):
        check_results = await asyncio.gather(*[p.check(str(p)) for p in particles])
        for p, check_result in zip(particles, check_results):
            p.check_result = check_result

    if visualize:
        # Figure out path to save JSON.
        if visualization_dir is None:
            json_path = json_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            json_relative = (
                json_file
                if json_file is not None
                else f"{model.__class__.__name__}-{timestamp}.json"
            )
            json_path = f"{visualization_dir}/{json_file}"

        # Save JSON
        with open(json_path, "w") as f:
            f.write(history.to_json())

        # Web path is the part of the path after the html directory
        if visualization_dir is not None:
            print(f"Visualize at http://localhost:8000/smc.html?path={json_relative}")
        else:
            print(f"Saved record to {json_path}")

    return particles
