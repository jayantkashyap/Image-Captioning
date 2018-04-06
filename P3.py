from train import train, predict

def P3():
    print("hello")
    while(True):
        choice = input("Enter choice: \n\t1. Train\n\t2. Predict")
        if choice == 1:
            train()
        elif choice == 2:
            image = input("Enter Image name: ")
            predict(image)
        else:
            print("Enter a valid option!")

        yn = input("Do you want to continue?")
        if yn == 'y' or yn == 'Y':
            continue
        elif yn == 'n' or yn == 'N':
            break
        else:
            print("Enter a valid option!")

if __name__ == '__main__':
    P3()