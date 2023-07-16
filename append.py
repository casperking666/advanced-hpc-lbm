with open('commands.txt', 'w') as f:
    for i in range(1, 113):
        command = f"mpirun -np {i} ./d2q9-bgk-reduce-v2 input_1024x1024.params obstacles_1024x1024.dat\n"
        f.write(command)
