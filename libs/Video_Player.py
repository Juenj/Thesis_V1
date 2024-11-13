import vlc
import time
import tkinter as tk
from tkinter import Button, Label
import numpy as np

class VideoPlayer:
    def __init__(self, video_path, video_offset=0, distance_index_list=None):
        self.video_path = video_path
        self.video_offset = video_offset
        self.distance_index_list = distance_index_list or []
        self.current_index = 0

        # Create a VLC instance and media player
        self.vlc_instance = vlc.Instance()
        self.player = self.vlc_instance.media_player_new()
        self.media = self.vlc_instance.media_new(video_path)
        self.player.set_media(self.media)

        # Initialize the Tkinter GUI
        self.root = tk.Tk()
        self.root.title("VLC Video Control")

        # Set up the UI
        self.setup_ui()

        # Load the video and pause it initially
        self.player.play()
        time.sleep(2)
        self.player.pause()

    def setup_ui(self):
        # Pause Button
        pause_button = Button(self.root, text="Pause", command=self.pause_video)
        pause_button.pack(side=tk.LEFT)

        # Play Button
        play_button = Button(self.root, text="Play", command=self.play_video)
        play_button.pack(side=tk.LEFT)

        # Next Button
        next_button = Button(self.root, text="Next", command=self.next_time)
        next_button.pack(side=tk.LEFT)

        # Previous Button
        previous_button = Button(self.root, text="Previous", command=self.previous_time)
        previous_button.pack(side=tk.LEFT)

        # Close Button
        close_button = Button(self.root, text="Close", command=self.close_video)
        close_button.pack(side=tk.LEFT)

        # Time Label
        self.time_label = Label(self.root, text="Time: 0 seconds")
        self.time_label.pack(side=tk.LEFT)

    def seek_to_time(self, seconds):
        """Seek to a specific time in the video."""
        if isinstance(seconds, (int, float)):
            self.player.set_time(int((seconds + self.video_offset) * 1000))

    def pause_video(self):
        """Pause the video."""
        self.player.pause()

    def play_video(self):
        """Resume the video."""
        self.player.play()

    def close_video(self):
        """Close the video properly and destroy the Tkinter window."""
        if self.player.is_playing():
            self.player.stop()
        self.player.release()
        self.root.destroy()

    def update_time_label(self):
        """Update the current time in the video player."""
        current_time = self.player.get_time() // 1000  # Get time in seconds
        self.time_label.config(text=f"Time: {current_time} seconds")
        self.root.after(1000, self.update_time_label)

    def next_time(self):
        """Skip to the next closest situation time."""
        if self.distance_index_list:
            self.current_index = (self.current_index + 1) % len(self.distance_index_list)
            specific_time_in_seconds = self.distance_index_list[self.current_index][0]
            self.seek_to_time(specific_time_in_seconds)
            self.time_label.config(text=f"Time: {specific_time_in_seconds} seconds")

    def previous_time(self):
        """Skip to the previous closest situation time."""
        if self.distance_index_list:
            self.current_index = (self.current_index - 1) % len(self.distance_index_list)
            specific_time_in_seconds = self.distance_index_list[self.current_index][0]
            self.seek_to_time(specific_time_in_seconds)
            self.time_label.config(text=f"Time: {specific_time_in_seconds} seconds")

    def start(self, initial_time):
        """Set the initial time and start the Tkinter main loop."""
        print(f"Initial Time: {initial_time} seconds")
        self.seek_to_time(initial_time)
        self.update_time_label()
        self.root.mainloop()

