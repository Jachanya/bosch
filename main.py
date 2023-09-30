import numpy as np
import pandas as pd
import pygame
import time
import matplotlib.pyplot as plt
import pickle
from scipy.fft import fft, fftfreq

from pygame.locals import (
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)
from collections import deque

# Collect program data
df = pd.read_csv("program_data.csv")

def generate_data():
    for data in df.iterrows():
        yield data

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(18, 18))
fig.suptitle('Vertically stacked subplots')

ax1.plot(df.iloc[:,-1],df.iloc[:,-2] * 10)
ax1.plot(df.iloc[:, -1], df.iloc[:, 6]/ 128, label="x")
ax1.plot(df.iloc[:, -1], df.iloc[:, 7]/ 128, label="y")
ax1.set_title("Object 4")

ax2.plot(df.iloc[:,-1],df.iloc[:,-2] * 10)
ax2.plot(df.iloc[:, -1], df.iloc[:, 4]/ 128, label="x")
ax2.plot(df.iloc[:, -1], df.iloc[:, 5]/ 128, label="y")
ax2.set_title("Object 3")

ax3.plot(df.iloc[:,-1],df.iloc[:,-2] * 10)
ax3.plot(df.iloc[:, -1], df.iloc[:, 2]/ 128, label="x")
ax3.plot(df.iloc[:, -1], df.iloc[:, 3]/ 128, label="y")
ax3.set_title("Object 2")

ax4.plot(df.iloc[:,-1],df.iloc[:,-2] * 10)
ax4.plot(df.iloc[:, -1], df.iloc[:, 0]/ 128, label="x")
ax4.plot(df.iloc[:, -1], df.iloc[:, 1]/ 128, label="y")
ax4.set_title("Object 1")

plt.legend()
plt.title("Yaw Rate")
plt.ylabel("Yaw")
plt.show()

# Define a perspective for object motion

def car_model(pos, velocity, yaw, time_change):
    return pos + (velocity * (np.cos(yaw)) * time_change)

def look_at(camera_position = np.array([0.0, 0.0,3.0]), camera_target = np.array([0.0, 0.0,0.0]), up = np.array([0.0, 1.0,0.0])):
    # Camera Direction
    camera_direction = (camera_position - camera_target) / np.linalg.norm((camera_position - camera_target))
    up_cam_direction = np.cross(up, camera_direction)
    camera_right = up_cam_direction / np.linalg.norm(up_cam_direction)
    camera_up = np.cross(camera_direction, camera_right)
    
    left_mat = np.array([
                        np.append(camera_right, [0]),
                        np.append(camera_up, [0]),
                        np.append(camera_direction, [0]),
                        np.array([0,0,0,1])
                          ])
    
    right_mat = np.array([[1,0,0, -camera_position[0]],
                         [0,1,0, -camera_position[1]],
                         [0,0,1, -camera_position[2]],
                         [0,0,0, 1]])
    
    return np.matmul(left_mat, right_mat)


# #Camera position
# camera_position = np.array([0.0, 0.0,3.0])
# # Camera Target
# camera_target = np.array([0.0, 0.0,0.0])
# # camera up and right vector
# camera_up = np.array([0.0, 1.0,0.0])
# look = look_at(camera_position, camera_target, camera_up)
# direction_x = np.cos(np.radians(yaw))
#  direction_z = np.sin(np.radians(yaw))
# camera_front = np.array([direction_x, 0, direction_y]) / np.linalg.norm(np.array([direction_x, 0, direction_y]))
# camera_target = np.array(camera_position + camera_front)

filename = "finalized_model.sav"
loaded_model = pickle.load(open(filename, 'rb'))
pygame.init()

# Set up the drawing window

def rotation_mat(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

def limits(value):
    return np.abs(value) > 1 and np.abs(value) < 5

SCREEN_CENTER_X = 100
SCREEN_CENTER_Y = 220

def plot(x, y, yaw, color = (255,0,0)):
    
    line_color = (25, 20, 230)
    
    rect = pygame.Surface((10, 10), pygame.SRCALPHA)
    rect = pygame.transform.rotate(rect, yaw)
    rect.fill(color)
    screen.blit(rect, pygame.Rect(x + SCREEN_CENTER_X, y + SCREEN_CENTER_Y, 10, 10).center)

    if (limits(x) and limits(y)):
        pygame.draw.circle(screen, (0, 0, 255), (10, 10), 10)
        pygame.draw.line(screen,line_color, (x + SCREEN_CENTER_X, y + SCREEN_CENTER_Y), 
                                            (SCREEN_CENTER_X, SCREEN_CENTER_Y))
        

def draw_rect(data, sequence_data):

    yaw = np.radians(data["YawRate"])
    direction_x = np.cos(yaw)
    direction_y = np.sin(yaw)
    
    #pygame.draw.rect(screen, color, pygame.Rect( x1,y1,
    #                                            OBJECT_SIZE, OBJECT_SIZE))
    plot(data["FirstObjectDistance_X"]/128, data["FirstObjectDistance_Y"]/128, yaw)
    plot(data["SecondObjectDistance_X"]/128, data["SecondObjectDistance_Y"]/128, yaw)
    plot(data["ThirdObjectDistance_X"]/128, data["ThirdObjectDistance_Y"]/128, yaw)
    plot(data["FourthObjectDistance_X"]/128, data["FourthObjectDistance_Y"]/128, yaw)
    plot(0, 0, yaw)

    # Predicting trajectory
    trajectory = []
    output = np.array([data.values])
    output[:, :8] /= 128
    output[:, 9:17] /= 256
    for i in range(10):
        output = loaded_model.predict(output)
        trajectory.append(output)
        
        pygame.draw.circle(screen, (0, 0, 255), 
                           (output[0][0] + SCREEN_CENTER_X, 
                            output[0][1]+ SCREEN_CENTER_Y) , 1)
        pygame.draw.circle(screen, (0, 0, 255), 
                           (output[0][2] + SCREEN_CENTER_X, 
                            output[0][3] + SCREEN_CENTER_Y), 1)
        pygame.draw.circle(screen, (0, 0, 255), 
                           (output[0][4] + SCREEN_CENTER_X, 
                            output[0][5] + SCREEN_CENTER_Y), 1)
        pygame.draw.circle(screen, (0, 0, 255), 
                           (output[0][6] + SCREEN_CENTER_X, 
                            output[0][7] + SCREEN_CENTER_Y), 1)
   


if __name__ == "__main__":
    screen = pygame.display.set_mode([500, 500])

    # Run until the user asks to quit
    running = True
    # Main loop
    sequence_car_motion = deque(maxlen=10)
    for _, data in generate_data():
        sequence_car_motion.append(data)
        # for loop through the event queue
        for event in pygame.event.get():
            # Check for KEYDOWN event
            if event.type == KEYDOWN:
                # If the Esc key is pressed, then exit the main loop
                if event.key == K_ESCAPE:
                    running = False
            # Check for QUIT event. If QUIT, then set running to false.
            elif event.type == QUIT:
                running = False

        # Fill the background with white
        screen.fill((255, 255, 255))
        color = (255,0,0)
        # Draw a solid blue circle in the center
        

        draw_rect(data, sequence_car_motion)
        # Flip the display
        pygame.display.flip()

    # Done! Time to quit.
    pygame.quit()
    print(df.columns)