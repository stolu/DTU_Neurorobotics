import pandas as pd
import matplotlib.pyplot as plt

def get_gdf_freq_data(file_path, bucket_length = 100):
    '''Input:
        file_path: path to GDF file
        bucket_length: buckets in ms, that are analysed individually
    Output:
        data dict with keys
            "center_time": middle of bucket in time
            "bucket_start_time": start time of bucket
            "bucket_end_time": end time of bucket
            "avg_cell_spike_freq": average cell spike frequency during bucket
            "unique_active_cells": Number of unique cells that spiked in bucket
            "spike_count": how many spikes during the bucket
    '''

    spike_time_list, population_id = get_sorted_gdf_data(file_path)
    total_cells = len(list(set(population_id)))

    bucket_spike_count = [0]
    last_time = 0

    spike_data = {  "center_time":[],
                    "bucket_start_time": [],
                    "bucket_end_time":[],
                    "avg_cell_spike_freq": [],
                    "unique_active_cells":[],
                    "spike_count":[]
                 }

    until = spike_time_list[0] + bucket_length
    earlier_spikes = 0
    for spike_count, spike_time in enumerate(spike_time_list):

        if spike_time > until:
            spikes_in_current_bucket = spike_count - earlier_spikes

            spike_data["center_time"].append( spike_time - (bucket_length/2.) )
            spike_data["bucket_start_time"].append( until - bucket_length )
            spike_data["bucket_end_time"].append( until)

            # Count number of unique cells in bucket
            pop_list = population_id[ earlier_spikes : spike_count]
            total_active_cells = len(pop_list)
            unique_active_cells = len(list(set(pop_list)))

            spike_data["spike_count"].append( spikes_in_current_bucket )
            spike_data["avg_cell_spike_freq"].append( (1000. * spikes_in_current_bucket) / (bucket_length * total_cells) )
            spike_data["unique_active_cells"].append(unique_active_cells)

            #print "Spikes in bucket:", spikes_in_current_bucket
            #print "Unique active cells:", unique_active_cells
            #print "Avg spike freq:", spike_data["avg_cell_spike_freq"][-1]

            until += bucket_length
            earlier_spikes = spike_count

    return spike_data



def get_sorted_gdf_data(gdf_filename):
    with open(gdf_filename, 'r') as data:
        population = []
        time = []
        for line in data:
            p = line.split()
            population.append(int(p[0]))
            time.append(float(p[1]))

        #Order both lists depending on the order of the time
        try:
            time, population = zip(*sorted(zip(time, population)))

        except Exception as e:
            print("ERROR: Error reading file: " + gdf_filename + " " + str(e) + "\n")
            time = [1.0]
            population = [-1]

    return time, population




def plot_gdf_freq_evolution(file_path, bucket_length = 100):

    spike_data = get_gdf_freq_data(file_path, bucket_length)
    df = pd.DataFrame(data=spike_data)

    fig_size = (10,5)#(14, 8)
    fig_dpi = 100#300

    freq_max = df["avg_cell_spike_freq"].max()
    freq_min = df["avg_cell_spike_freq"].min()

    fig = plt.figure(figsize=fig_size)#(figsize=fig_size, dpi=fig_dpi)
    ax = fig.gca()

    df.plot(x="bucket_start_time",#x="center_time",
            y="avg_cell_spike_freq",
            label="Avg Cell Spike Freq",
            kind="line",#"bar",
            color="b",
            ax=ax)#, style='o', ms=3)

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Frequency [Hz]")
    #ax.set_xlim(3,3.2)
    ax.set_ylim(freq_min * 0.999,freq_max * 1.001)
    plt.title(file_path.split("/")[-1] + " (bucket length: " + str(bucket_length) + "ms)")
    plt.grid()
    plt.show()

    #plt.show()
