import io
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

import tracking_dl


st.set_page_config(page_title="Ultrasound Tracking Demo", layout="wide")


@st.cache_resource
def load_model(device):
    model = tracking_dl.ConvLSTMTracker().to(device)
    return model


def train_model(model, device, num_epochs, batch_size, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    epoch_losses = []

    for _ in range(num_epochs):
        losses = []
        for _ in range(6):
            batch_inputs = []
            batch_targets = []
            for _ in range(batch_size):
                inputs, targets = tracking_dl.generate_sequence(
                    num_frames=10,
                    image_size=(64, 64),
                    num_bubbles=5,
                )
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

        epoch_losses.append(float(np.mean(losses)))

    return epoch_losses


def render_gif(inputs, heatmaps):
    images = []
    for t in range(heatmaps.shape[0]):
        coords = tracking_dl.detect_peaks(heatmaps[t])

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_title(f"Frame {t + 1} - DL Tracking")
        ax.imshow(inputs[t], cmap="gray")
        if len(coords) > 0:
            ax.scatter(coords[:, 1], coords[:, 0], color="cyan", marker="o", facecolors="none")
        fig.canvas.draw()

        image = np.asarray(fig.canvas.buffer_rgba())
        images.append(image)
        plt.close(fig)

    buffer = io.BytesIO()
    imageio.mimsave(buffer, images, fps=2, format="GIF")
    buffer.seek(0)
    return buffer


def main():
    st.title("Ultrasound Microbubble Tracking - DL Demo")

    with st.sidebar:
        st.header("Settings")
        num_frames = st.slider("Frames", 6, 20, 12)
        num_bubbles = st.slider("Bubbles", 1, 12, 5)
        image_size = st.selectbox("Image size", [64, 96, 128], index=0)
        num_epochs = st.slider("Train epochs", 0, 6, 2)
        batch_size = st.slider("Batch size", 1, 8, 4)
        lr = st.select_slider("Learning rate", options=[1e-4, 5e-4, 1e-3, 2e-3], value=1e-3)
        run_button = st.button("Run demo")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    if run_button:
        if num_epochs > 0:
            with st.spinner("Training..."):
                losses = train_model(model, device, num_epochs, batch_size, lr)
            st.success("Training complete")
            st.line_chart({"loss": losses})

        inputs, _ = tracking_dl.generate_sequence(
            num_frames=num_frames,
            image_size=(image_size, image_size),
            num_bubbles=num_bubbles,
        )
        input_tensor = torch.from_numpy(inputs[None, :, None, :, :]).to(device)

        with torch.no_grad():
            outputs = model(input_tensor).cpu().numpy()[0, :, 0]

        gif_buffer = render_gif(inputs, outputs)
        st.image(gif_buffer.getvalue(), caption="DL tracking result")
        st.download_button(
            label="Download GIF",
            data=gif_buffer.getvalue(),
            file_name="bubble_tracking_dl_demo.gif",
            mime="image/gif",
        )


if __name__ == "__main__":
    main()
