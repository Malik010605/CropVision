# CropVision Recording Guide

## Recommended deliverable

- Resolution: **1920 × 1080**, 16:9, 30 frames per second.
- Main content: full-screen slide presentation or short live demonstration.
- Webcam: current speaker visible in the **top-right corner** throughout their section.
- Audio: one clear voice at a time, no music under speech.
- Final duration: target **9:30** and remain strictly below **10:00**.

The slides intentionally reserve the upper-right area. Keep the webcam in that same location for every speaker and do not resize it between sections.

## Option A — OBS Studio, recommended

### Create one shared scene profile

1. Set **Settings → Video → Base Canvas** and **Output Resolution** to `1920 × 1080`, with `30 FPS`.
2. Add a **Display Capture** or **Window Capture** source for the slide show. Fit it to the full canvas.
3. Add a **Video Capture Device** for the current speaker.
4. Crop the camera to 16:9 and place it at approximately `x = 1460`, `y = 40`, with a size near `430 × 242` pixels. Add a thin white border or subtle shadow if useful.
5. Add the speaker's microphone as an **Audio Input Capture**. Disable duplicate desktop or laptop microphones.
6. Use H.264 recording at roughly **8–12 Mb/s** and AAC audio at **48 kHz, 192 kb/s**. Record to MKV for crash safety, then use **File → Remux Recordings** to produce MP4.

### Match audio between speakers

- Record in a quiet room with the microphone 15–25 cm from the speaker.
- Aim for normal speech around **−18 to −12 dBFS**, with peaks below **−6 dBFS** in OBS.
- Use the same microphone when possible. Otherwise, run a short test and match perceived loudness before recording.
- During editing, normalize the complete program to about **−16 LUFS integrated**, with a true peak no higher than **−1 dB**.

## Three-speaker workflow

Use the same PowerPoint file, OBS scene collection, webcam size, and output settings for all three recordings.

1. **Edgar** records slides 1–4. Start with two seconds of silence, speak from 0:00 to 3:10, and hold slide 4 for two seconds after the handoff sentence.
2. **Malik** records slides 5–7. Leave two silent seconds at both ends. The edited section starts at 3:10 and ends at 6:20.
3. **Etienne** records slides 8–10. Leave two silent seconds at both ends. The edited section starts at 6:20 and ends at 9:30.
4. Each student should look at the camera for their first and last sentence. During charts, glance toward the shared content but keep the face visible.
5. Upload the original recordings without messaging-app compression.

### Combine the sections

DaVinci Resolve, iMovie, CapCut, or another timeline editor can combine the three MP4 files.

- Create a `1920 × 1080`, 30 fps timeline.
- Trim the silent handles and place Edgar at 0:00, Malik at 3:10, and Etienne at 6:20.
- Use a short 0.15–0.25 second dissolve, or a direct cut on the spoken handoff. Do not add long transition cards that push the video over ten minutes.
- Check that the slide changes follow `Presentation_Script.md` and that the camera remains in the same top-right position.
- Match audio loudness, remove obvious background noise carefully, and listen once with headphones.
- Export H.264 MP4 at 1080p. Confirm the exported duration is below 10:00 and watch the complete file before submission.

## Option B — Zoom recording

Zoom is simpler but provides less precise layout control.

1. Each student shares only the PowerPoint slide-show window.
2. Enable the speaker video and select **Record video during screen sharing**.
3. Keep the floating video panel in the upper-right corner and use the same size for every recording.
4. Disable notifications, meeting controls, and participant thumbnails before recording.
5. Record each section separately, then combine the files using the timeline instructions above.

If Zoom moves or hides the camera overlay in the recorded file, use OBS instead.

## Optional demonstration

The professor allows slides and/or a demonstration. The ten-slide version already explains the complete project and fits the target duration. If a demonstration is added, replace part of the narration rather than extending the video. A safe demonstration is a brief view of the GitHub repository, `SETUP.md`, and saved results; do not rerun training or final evaluation during the recording.

## Final quality checklist

- [ ] Total duration is under 10:00; target is 9:30.
- [ ] Edgar, Malik, and Etienne each speak for approximately 3:10.
- [ ] The current speaker's face is visible in the upper-right corner.
- [ ] The webcam never covers important slide text, figures, axes, or labels.
- [ ] Slides fill the screen and remain readable at 1080p.
- [ ] Audio is clear, consistent, and free from clipping.
- [ ] Handoffs at 3:10 and 6:20 are smooth.
- [ ] The final test results are identified as test results, not validation results.
- [ ] The finished MP4 has been watched from beginning to end.

