!/usr/bin/bash
# while [1]; do
xdotool mousemove 2643 638 click 1 
sleep 0.5
xdotool mousemove 2649 906 click 1
sleep 1
xdotool mousemove 2953 735 click 1 
sleep 0.5
rosrun f1rl spawner.py
sleep 2
echo "Done"