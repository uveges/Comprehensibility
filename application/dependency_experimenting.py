import subprocess

if __name__ == '__main__':
    magyarlanc_path = "/home/istvanu/PycharmProjects/comprehensibility_FULL/resources/magyarlanc-3.0.jar"
    input = "/home/istvanu/PycharmProjects/comprehensibility_FULL/application/ml_folder/test.txt"
    output = input.replace(".txt", ".out")

    args = ["java", "-Xmx2G", "-jar", magyarlanc_path, "-mode", "depparse", "-input", input, "-output", output]

    subprocess.call(args)

    with open(output, 'r', encoding='utf8') as conll:
        lines = conll.readlines()

    lines = [line.replace('\\t', '\t').replace('\\n', '\n') for line in lines]

    for line in lines:
        print(line)