from time import sleep


if __name__ == '__main__':
    # sleep(5)
    with open("output/results.csv", "w+") as f:
        f.write("Results!")
