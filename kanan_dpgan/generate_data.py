import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt

# Function to generate synthetic SEIHR-style data
def generate_synthetic_seihr():
    time = np.arange(100)
    susceptible = 1000 - 5 * time
    exposed = 5 * time - 2 * np.sin(time)
    infected = 10 * np.sin(time / 5) + 50
    hospitalized = 3 * np.cos(time / 4) + 10
    recovered = 5 * time + 30

    df = pd.DataFrame({
        'time': time,
        'susceptible': susceptible.clip(min=0),
        'exposed': exposed.clip(min=0),
        'infected': infected.clip(min=0),
        'hospitalized': hospitalized.clip(min=0),
        'recovered': recovered.clip(min=0),
    })
    
    df.to_csv("time_series.csv", index=False)
    print("Data saved to time_series.csv")
    return df

if __name__ == "__main__":
    df = generate_synthetic_seihr()
    df.plot(x='time')
    plt.title("Simulated Time Series Data")
    plt.show()
    input("Press Enter to exit...")
