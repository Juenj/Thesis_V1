from mplsoccer import Pitch
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from libs.weight_generator import *

class InteractivePitch:
    def __init__(self, match_data):
        self.match_data = match_data  # Store match data
        # Initialize pitch
        self.football_pitch = Pitch(
            pitch_type='skillcorner', pitch_length=105, pitch_width=68,
            axis=True, label=True, line_color="white", pitch_color="grass"
        )
        self.fig, self.ax = self.football_pitch.draw(figsize=(10, 7))
        
        # Data structures for storing points, vectors, situations, and ball position
        self.points = []
        self.vectors = []
        self.situations = []
        self.ball_position = None
        
        # Mode flags
        self.draw_vector_mode = False
        self.place_ball_mode = False
        
        # Initialize player dropdowns
        self._initialize_players(match_data)
        
        # Set up UI elements
        self._setup_ui()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
    
    
    def on_click(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            if self.place_ball_mode:
                self.ball_position = (x, y)
                self.ax.plot(x, y, 'o', color='green', markersize=10, label="Ball")
                plt.draw()
            elif self.draw_vector_mode:
                if self.vector_start is None:
                    self.vector_start = (x, y)
                    self.ax.plot(x, y, 'bo')
                else:
                    vector_end = (x, y)
                    self.ax.annotate('', xy=vector_end, xytext=self.vector_start,
                                     arrowprops=dict(facecolor='red', shrink=0.05))
                    self.vectors.append((self.vector_start, vector_end))
                    self.vector_start = None
            else:
                self.points.append((x, y))
                self.ax.plot(x, y, 'ro')

    def save_situation(self, _):
        if self.points or self.vectors or self.ball_position:
            self.situations.append({'points': list(self.points), 'vectors': list(self.vectors), 'ball': self.ball_position})
            self._update_situation_dropdown()
            print(f"Situation saved! Total saved situations: {len(self.situations)}")
    
    def calculate_wasserstein(self, _):
        if self.situations and self.ball_position:
            # Retrieve selected function from the dropdown
            selected_function = self.function_dropdown.value
            weighting_function = {
                "control": lambda x: 1,
                "function_1": lambda x: 200 - x,
                "function_2": lambda x: 1 / x,
                "function_3": lambda x: np.exp(-x / 40)
            }[selected_function]
            
            # Prepare clicked row from the last saved situation
            clicked_situation = self.situations[-1]  # Use the most recently saved situation
            clicked_row = self._situation_to_row(clicked_situation)
            
            # Calculate Wasserstein distances
            indices = most_similar_with_wasserstein_from_row(clicked_row, self.match_data, weighting_function)
            print("Wasserstein calculated, closest situations:", indices[:10])  # Display the top 10 closest situations

    def _situation_to_row(self, situation):
        """Convert a saved situation (points and ball position) to a 1D row format compatible with the DataFrame."""
        row = {}
        for i, (x, y) in enumerate(situation['points']):
            row[f'home_{i + 1}_x'] = x
            row[f'home_{i + 1}_y'] = y
        if situation['ball']:
            row['ball_x_team'] = situation['ball'][0]
            row['ball_y_team'] = situation['ball'][1]
        return row
    
    def _initialize_players(self, match_data):
        """Initialize home and away player lists from match data."""
        # Extract player numbers using regex and keep backups
        self.home_player_numbers = list(np.unique([player[:-1] for player in match_data.filter(regex="^home").columns.to_numpy()]))
        self.backup_home_player_numbers = self.home_player_numbers.copy()
        
        self.away_player_numbers = list(np.unique([player[:-1] for player in match_data.filter(regex="^away").columns.to_numpy()]))
        self.backup_away_player_numbers = self.away_player_numbers.copy()
        
        # Dropdowns for selecting players
        self.home_players_dropdown = widgets.Dropdown(
            options=[("Select Player", "")] + [(f"{player}", player) for player in self.home_player_numbers],
            description='Home Player:',
            disabled=False,
        )
        
        self.away_players_dropdown = widgets.Dropdown(
            options=[("Select Player", "")] + [(f"{player}", player) for player in self.away_player_numbers],
            description='Away Player:',
            disabled=False,
        )
        
        # Observe dropdown changes
        self.home_players_dropdown.observe(self.home_player_selected, names='value')
        self.away_players_dropdown.observe(self.away_player_selected, names='value')

    def on_click(self, event):
        if event.inaxes:  # Check if click is inside plot
            x, y = event.xdata, event.ydata  # Get coordinates
            
            if self.place_ball_mode:
                # Place the ball at the clicked position
                self.ball_position = (x, y)
                self.ax.plot(x, y, 'o', color='green', markersize=10, label="Ball" if not any(artist.get_label() == "Ball" for artist in self.ax.get_children()) else "")
                plt.draw()  # Update the plot
                print(f"Ball placed at: {self.ball_position}")

            elif self.draw_vector_mode:
                if self.vector_start is None:  # If no start point, set this as start point
                    self.vector_start = (x, y)
                    self.ax.plot(x, y, 'bo')  # Mark the start point with a blue dot
                else:
                    # If there's already a start point, draw the vector from start to this point
                    vector_end = (x, y)
                    self.ax.annotate('', xy=vector_end, xytext=self.vector_start,
                                     arrowprops=dict(facecolor='red', shrink=0.05))  # Draw vector
                    self.vectors.append((self.vector_start, vector_end))  # Save the vector
                    self.vector_start = None  # Reset the start point
            else:
                self.points.append((x, y))  # Add to list of points
                self.ax.plot(x, y, 'ro')  # Plot the point

    def save_situation(self, _):
        """Save the current situation of points, vectors, and ball position."""
        if self.points or self.vectors or self.ball_position:
            self.situations.append({'points': list(self.points), 'vectors': list(self.vectors), 'ball': self.ball_position})
            self._update_situation_dropdown()  # Update dropdown after saving
            print(f"Situation saved! Total saved situations: {len(self.situations)}")
        else:
            print("No players, vectors, or ball to save!")

    def clear_situation(self, _):
        """Clear the current situation and reset UI elements."""
        self.points = []
        self.vectors = []
        self.players = []
        self.vector_start = None
        self.ball_position = None

        # Reset player lists
        self.home_player_numbers = self.backup_home_player_numbers.copy()
        self.away_player_numbers = self.backup_away_player_numbers.copy()
        
        self._update_player_dropdowns()
        
        # Clear plot and redraw pitch
        self.ax.cla()
        self.football_pitch.draw(ax=self.ax)
        plt.draw()
        print("Cleared the current situation. All players are available for selection again.")

    def toggle_draw_vector(self, _):
        """Toggle vector drawing mode."""
        self.draw_vector_mode = not self.draw_vector_mode
        self.place_ball_mode = False
        if self.draw_vector_mode:
            print("Vector drawing mode enabled. Select start and end points for the vector.")
        else:
            print("Switched to player drawing mode.")

    def toggle_place_ball(self, _):
        """Toggle ball placement mode."""
        self.place_ball_mode = not self.place_ball_mode
        self.draw_vector_mode = False
        if self.place_ball_mode:
            print("Ball placement mode enabled. Click to place the ball on the pitch.")
        else:
            print("Ball placement mode disabled.")

    def select_player(self, player_num, dropdown, player_list):
        """Select and add player, remove from dropdown."""
        self.players.append(player_num)
        player_list.remove(player_num)
        self._update_dropdown_options(dropdown, player_list)

    def remove_player(self, player_num, dropdown, player_list):
        """Remove player from list and add back to dropdown."""
        if player_num in self.players:
            self.players.remove(player_num)
            player_list.append(player_num)
            player_list.sort()
            self._update_dropdown_options(dropdown, player_list)

    def _update_dropdown_options(self, dropdown, player_list):
        """Helper to update dropdown options based on player list."""
        dropdown.options = [("Select Player", "")] + [(f"{player}", player) for player in player_list]
        
    def _update_player_dropdowns(self):
        """Reset player dropdowns to full lists."""
        self._update_dropdown_options(self.home_players_dropdown, self.home_player_numbers)
        self._update_dropdown_options(self.away_players_dropdown, self.away_player_numbers)

    def _update_situation_dropdown(self):
        """Update the situation dropdown with the latest saved situations."""
        self.situation_dropdown.options = [("Select Situation", "")] + [(f"Situation {i+1}", i) for i in range(len(self.situations))]

    def home_player_selected(self, change):
        selected_player = change['new']
        if selected_player:
            self.select_player(selected_player, self.home_players_dropdown, self.home_player_numbers)

    def away_player_selected(self, change):
        selected_player = change['new']
        if selected_player:
            self.select_player(selected_player, self.away_players_dropdown, self.away_player_numbers)

    def load_situation(self, change):
        """Load a saved situation and plot it on the pitch."""
        selected_situation = change['new']
        if selected_situation != "":
            situation = self.situations[selected_situation]
            self.clear_situation(None)  # Clear the current plot first

            # Plot the points
            for x, y in situation['points']:
                self.ax.plot(x, y, 'ro')
                self.points.append((x, y))
            
            # Plot the vectors
            for start, end in situation['vectors']:
                self.ax.annotate('', xy=end, xytext=start, arrowprops=dict(facecolor='red', shrink=0.05))
                self.vectors.append((start, end))
            
            # Plot the ball if it exists
            if situation['ball']:
                bx, by = situation['ball']
                self.ax.plot(bx, by, 'o', color='green', markersize=10, label="Ball")
                self.ball_position = (bx, by)
            
            plt.draw()
            print(f"Loaded Situation {selected_situation + 1}")

    
    def _setup_ui(self):
        # Buttons and dropdowns
        self.save_button = widgets.Button(description="Save Situation", button_style='success')
        self.clear_button = widgets.Button(description="Clear", button_style='warning')
        self.toggle_vector_button = widgets.Button(description="Toggle Draw Vector", button_style='info')
        self.toggle_ball_button = widgets.Button(description="Place Ball", button_style='primary')
        
        # Function selection dropdown for Wasserstein
        self.function_dropdown = widgets.Dropdown(
            options=[("Control (1)", "control"), 
                     ("200 - x", "function_1"),
                     ("1 / x", "function_2"),
                     ("exp(-x / 40)", "function_3")],
            description='Wasserstein Function:'
        )
        
        # Calculate button for Wasserstein
        self.calculate_wasserstein_button = widgets.Button(description="Calculate Wasserstein", button_style='info')
        
        # Situation dropdown to reload saved situations
        self.situation_dropdown = widgets.Dropdown(description="Saved Situations:")
        self.situation_dropdown.observe(self.load_situation, names='value')
        
        # Connect button events
        self.save_button.on_click(self.save_situation)
        self.clear_button.on_click(self.clear_situation)
        self.toggle_vector_button.on_click(self.toggle_draw_vector)
        self.toggle_ball_button.on_click(self.toggle_place_ball)
        self.calculate_wasserstein_button.on_click(self.calculate_wasserstein)

        # Display UI layout
        display(widgets.HBox([
            widgets.VBox([self.home_players_dropdown, self.away_players_dropdown, self.situation_dropdown, self.function_dropdown]), 
            widgets.VBox([self.save_button, self.clear_button, self.toggle_vector_button, self.toggle_ball_button, self.calculate_wasserstein_button])
        ]))


    def display_ui(self):
        # Call this method to display the interactive UI in a notebook
        display(self.fig)
