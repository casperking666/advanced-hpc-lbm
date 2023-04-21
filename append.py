with open('commands.txt', 'w') as f:
    for i in range(1, 113):
        command = f"mpirun -np {i} ./d2q9-bgk-vectorized input_128x128.params obstacles_128x128.dat\n"
        f.write(command)
