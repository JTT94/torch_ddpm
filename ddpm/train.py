from tqdm import tqdm
from .utils import repeater


def train(model, diffusion, optimizer, dataloader, num_iterations):
    repeat_data_iter = repeater(dataloader)

    for it in tqdm(range(num_iterations)):
        (batch_x,) = next(repeat_data_iter)
        t = diffusion.sample_t(batch_x)
        perturbed_sample = diffusion.sample_x(batch_x, t)

        model_out = model(
            x=perturbed_sample.x_t, t=perturbed_sample.t.reshape(batch_x.shape[0], 1)
        )

        optimizer.zero_grad()

        loss = diffusion.loss(model_out, perturbed_sample.z)

        loss.backward()

        optimizer.step()
