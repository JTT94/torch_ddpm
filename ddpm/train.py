from tqdm import tqdm
from ddpm.utils import repeater


def train(model, diffusion, optimizer, dataloader, num_iterations, device):
    repeat_data_iter = repeater(dataloader)

    for it in tqdm(range(num_iterations)):
        batch_x = next(repeat_data_iter)[0].to(device)
        t = diffusion.sample_t(batch_x)
        perturbed_sample = diffusion.sample_x(batch_x, t)
        x_t = perturbed_sample.x_t
        model_out = model(x_t, t)

        optimizer.zero_grad()

        loss = diffusion.loss(model_out, perturbed_sample.z)

        loss.backward()

        optimizer.step()
