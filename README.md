# 🤚 Hand Gesture Controller

A real-time computer vision application that transforms your hand gestures into computer controls. Control your mouse, scroll, click, and execute keyboard shortcuts using intuitive hand movements captured through your webcam.

## 🌟 Features

### 🖱️ **Right Hand Controls**
- **Cursor Movement**: Move your hand to control the mouse cursor
- **Click Gesture**: Pinch thumb and index finger to click
- **Smart Click Prevention**: Prevents accidental clicks during fast movements
- **Smooth Movement**: One Euro Filter for fluid cursor motion

### 📜 **Left Hand Controls**
- **Scroll Gesture**: Pinch and drag up/down to scroll
- **Visual Feedback**: Shows scroll anchor point and movement

### 🤝 **Two-Hand Gestures**
- **Voice Typing**: Right index finger + Left pinky finger to activate Windows voice typing (`Win + H`)
- **Ctrl+Enter**: Right index finger + Left thumb to execute Ctrl+Enter shortcut

### ⏸️ **System Controls**
- **Pause/Resume**: Show the back of your left hand with all fingers spread and extended to pause/unpause the system
- **Visual Indicators**: Clear feedback for pause state and gesture recognition
- **Debug Mode**: Toggle visual debugging with FPS counter

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Webcam
- Windows (for voice typing feature)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vansio3/gesture-contoller.git
   cd gesture-contoller
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

## 📋 Usage Guide

### Getting Started
1. Position yourself in front of your webcam
2. Run the application
3. Hold your right hand up to start controlling the cursor
4. Use the gestures described below

### Gesture Controls

| Gesture | Action | Description |
|---------|---------|-------------|
| **Right Hand Move** | Cursor Movement | Move your right hand to control the mouse |
| **Right Pinch** | Left Click | Touch thumb and index finger to click |
| **Left Pinch + Drag** | Scroll | Pinch left hand and drag up/down |
| **Right Index + Left Pinky** | Voice Typing | Bring fingers close to activate voice input |
| **Right Index + Left Thumb** | Ctrl+Enter | Bring fingers close to execute shortcut |
| **Left Hand Back (Spread Fingers)** | Pause/Resume | Hold for 2 seconds to toggle system state |

### Keyboard Shortcuts
- **Q**: Quit application
- **D**: Toggle debug mode (shows FPS and visual feedback)

## ⚙️ Configuration

The application includes several configurable parameters in `app.py`:

### Movement Settings
```python
HORIZONTAL_INPUT_START = 0.4    # Left boundary of gesture area
HORIZONTAL_INPUT_END = 0.9      # Right boundary of gesture area
VERTICAL_TOP_MARGIN = 0.5       # Top boundary of gesture area
VERTICAL_BOTTOM_MARGIN = 0.8    # Bottom boundary of gesture area
DEAD_ZONE_RADIUS = 10.0         # Minimum movement before cursor moves
```

### Filter Settings
```python
config = {'min_cutoff': 0.8, 'beta': 2.0}  # One Euro Filter parameters
```

### Gesture Thresholds
```python
CLICK_THRESHOLD = 14            # Distance for click detection
SCROLL_THRESHOLD = 22           # Distance for scroll activation
VOICE_TYPING_THRESHOLD = 25     # Distance for voice typing activation
CTRL_ENTER_THRESHOLD = 25       # Distance for Ctrl+Enter activation
```

### Timing Settings
```python
CLICK_COOLDOWN = 2.0            # Prevent rapid clicking
VOICE_TYPING_COOLDOWN = 3.0     # Cooldown between voice typing activations
CTRL_ENTER_COOLDOWN = 3.0       # Cooldown between Ctrl+Enter activations
PAUSE_COOLDOWN = 2.0            # Cooldown between pause toggles
PAUSE_HOLD_DURATION = 2.0       # Time to hold pause gesture
```

## 🛠️ Technical Details

### Architecture
- **MediaPipe Hands**: Real-time hand landmark detection
- **OpenCV**: Computer vision and image processing
- **PyAutoGUI**: System control (mouse and keyboard)
- **One Euro Filter**: Smooth cursor movement filtering

### Performance
- Real-time processing with FPS monitoring
- Optimized for low-latency gesture recognition
- Configurable camera resolution (default: 640x480)

### Hand Detection
- Supports up to 2 hands simultaneously
- Minimum confidence thresholds for reliable detection
- Automatic handedness classification (left/right)

## 🔧 Troubleshooting

### Common Issues

**Camera not working**
- Ensure your webcam is connected and not being used by another application
- Check camera permissions in your operating system

**Gestures not recognized**
- Ensure good lighting conditions
- Keep hands clearly visible to the camera
- Adjust gesture thresholds if needed
- Try different distances from the camera

**Cursor movement is jerky**
- Adjust One Euro Filter parameters for smoother movement
- Increase `min_cutoff` value for more filtering
- Check CPU usage - close other applications if needed

**Clicks not working**
- Ensure you're extending both index and middle fingers when pinching
- Check that you're not moving too fast during the gesture
- Verify the click cooldown has elapsed

### Debug Mode
Press **D** during operation to toggle debug mode, which shows:
- Real-time FPS counter
- Visual gesture feedback
- Hand landmark detection
- Gesture state indicators

## 📁 Project Structure

```
gesture-controller/
├── app.py                 # Main application
├── one_euro_filter.py     # Smoothing filter implementation
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── build/                # Build artifacts
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional gesture recognition
- Performance optimization
- Cross-platform compatibility
- New control schemes
- Better filtering algorithms

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## ⚠️ Disclaimer

- This application controls your computer - use with caution
- The fail-safe is disabled (`pyautogui.FAILSAFE = False`) - be careful with automation
- Test gestures in a safe environment first
- Not responsible for any unintended actions or system changes

---

**Happy Gesturing!** 🎉
