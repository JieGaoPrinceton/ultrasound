import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import maximum_filter


def add_gaussian(image, center, sigma=1.5, amplitude=1.0):
    height, width = image.shape
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    cx, cy = center[1], center[0]
    gaussian = amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    image += gaussian


def generate_sequence(num_frames, image_size, num_bubbles, noise_std=0.05, sigma=1.5):
    height, width = image_size
    trajectories = [np.array([np.random.randint(0, height), np.random.randint(0, width)]) for _ in range(num_bubbles)]
    frames = []

    for _ in range(num_frames):
        frame = np.zeros((height, width), dtype=np.float32)
        new_positions = []
        for pos in trajectories:
            move = np.random.randint(-2, 3, size=2)
            new_pos = np.clip(pos + move, [0, 0], [height - 1, width - 1])
            new_positions.append(new_pos)
            add_gaussian(frame, new_pos, sigma=sigma, amplitude=1.0)
        trajectories = new_positions
        frames.append(frame)

    frames = np.stack(frames, axis=0)
    noisy_frames = frames + noise_std * np.random.randn(*frames.shape).astype(np.float32)
    return noisy_frames, frames


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMTracker(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=16):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels)
        self.out_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        batch_size, seq_len, _, height, width = x.shape
        h = torch.zeros(batch_size, self.cell.hidden_channels, height, width, device=x.device)
        c = torch.zeros_like(h)
        outputs = []

        for t in range(seq_len):
            h, c = self.cell(x[:, t], h, c)
            out = self.out_conv(h)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)
        return outputs


def detect_peaks(heatmap, threshold_factor=1.5, size=7):
    threshold = heatmap.mean() + threshold_factor * heatmap.std()
    local_max = (heatmap == maximum_filter(heatmap, size=size)) & (heatmap > threshold)
    coords = np.argwhere(local_max)
    return coords


def train_demo(model, device, num_epochs=3, batch_size=4):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(num_epochs):
        losses = []
        for _ in range(10):
            batch_inputs = []
            batch_targets = []
            for _ in range(batch_size):
                inputs, targets = generate_sequence(num_frames=10, image_size=(64, 64), num_bubbles=5)
                batch_inputs.append(inputs)
                batch_targets.append(targets)
            batch_inputs = torch.from_numpy(np.stack(batch_inputs)[:, :, None, :, :]).to(device)
            batch_targets = torch.from_numpy(np.stack(batch_targets)[:, :, None, :, :]).to(device)

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"Epoch {epoch + 1}: loss={np.mean(losses):.4f}")


def run_inference(model, device, output_path="bubble_tracking_dl.gif"):
    model.eval()
    inputs, targets = generate_sequence(num_frames=12, image_size=(64, 64), num_bubbles=5)
    input_tensor = torch.from_numpy(inputs[None, :, None, :, :]).to(device)

    with torch.no_grad():
        outputs = model(input_tensor).cpu().numpy()[0, :, 0]

    images = []
    for t in range(outputs.shape[0]):
        heatmap = outputs[t]
        coords = detect_peaks(heatmap)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_title(f"Frame {t + 1} - DL Tracking")
        ax.imshow(inputs[t], cmap="gray")
        if len(coords) > 0:
            ax.scatter(coords[:, 1], coords[:, 0], color="cyan", marker="o", facecolors="none")
        fig.canvas.draw()

        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        images.append(image)
        plt.close(fig)

    imageio.mimsave(output_path, images, fps=2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTMTracker().to(device)
    train_demo(model, device)
    run_inference(model, device)


if __name__ == "__main__":
    main()
