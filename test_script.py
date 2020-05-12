import sys
import subprocess
import csv
from os import stat

csv_header = ["MedSeq_time", "MedPar_time", "LinSeq_time", "LinPar_time", "SobXseq_time", "SobXpar_time", "SobYseq_time", "SobYpar_time", "SobEseq_time", "SobEpar_time", "No_threads"] 
outdir = "TestResults/"

def run_test(filename, threads):
    command = ("./406im", "cppinp/" + filename + ".txt", " cppout/" + filename, "0")
    medseq_times = []
    medpar_times =[]
    linseq_times = []
    linpar_times = []
    sobxseq_times = []
    sobxpar_times = []
    sobyseq_times = []
    sobypar_times = []
    sobeseq_times = []
    sobepar_times = []
    for i in range(10):
        popen = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        test_out = popen.stdout.read()
        out = test_out.decode('utf-8')
        for line in out.splitlines():
            if "Median Sequential" in line:
                medseq_times.append(float(line.split(":")[1]))
            elif "Median Parallel" in line:
                medpar_times.append(float(line.split(":")[1]))
            elif "LinScale Sequential" in line:
                linseq_times.append(float(line.split(":")[1]))
            elif "LinScale Parallel" in line:
                linpar_times.append(float(line.split(":")[1]))
            elif "SobelFilter(X) Sequential" in line:
                sobxseq_times.append(float(line.split(":")[1]))
            elif "SobelFilter(X) Parallel" in line:
                sobxpar_times.append(float(line.split(":")[1]))
            elif "SobelFilter(Y) Sequential" in line:
                sobyseq_times.append(float(line.split(":")[1]))
            elif "SobelFilter(Y) Parallel" in line:
                sobypar_times.append(float(line.split(":")[1]))
            elif "SobelEdge Sequential" in line:
                sobeseq_times.append(float(line.split(":")[1]))
            elif "SobelEdge Parallel" in line:
                sobepar_times.append(float(line.split(":")[1]))
    medseq = sum(medseq_times) / len(medseq_times)
    medpar = sum(medpar_times) / len(medpar_times)
    linseq = sum(linseq_times) / len(linseq_times)
    linpar = sum(linpar_times) / len(linpar_times)
    sobxseq = sum(sobxseq_times) / len(sobxseq_times)
    sobxpar = sum(sobxpar_times) / len(sobxpar_times)
    sobyseq = sum(sobyseq_times) / len(sobyseq_times)
    sobypar = sum(sobypar_times) / len(sobypar_times)
    sobeseq = sum(sobeseq_times) / len(sobeseq_times)
    sobepar = sum(sobepar_times) / len(sobepar_times)
    csv_name = outdir + filename + ".csv"
    with open(csv_name, "a+") as f:
        writer = csv.writer(f)
        if stat(csv_name).st_size == 0:
            writer.writerow(csv_header)
        writer.writerow([medseq, medpar, linseq, linpar, sobxseq, sobxpar, sobyseq, sobypar, sobeseq, sobepar, threads])
if __name__ == '__main__':
    filename = sys.argv[1]
    threads = sys.argv[2]
    run_test(filename, threads)
