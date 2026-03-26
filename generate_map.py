import folium
import pandas as pd

# Load station data
df = pd.read_csv("station_delay.csv")

# Create map centered on India
m = folium.Map(location=[23.5, 80.0], zoom_start=5)

# Color function based on delay
def get_color(delay):
    if delay < 20:
        return "green"
    elif delay < 40:
        return "orange"
    else:
        return "red"

# Add station markers
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lng"]],
        radius=10,
        popup=f"{row['station']}<br>Avg Delay: {row['avg_delay']} min",
        color=get_color(row["avg_delay"]),
        fill=True,
        fill_color=get_color(row["avg_delay"]),
    ).add_to(m)

# Save map
m.save("templates/heatmap.html")
print("Heatmap generated → templates/heatmap.html")
