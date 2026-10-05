import train
from code import Encode ,Decode

def main():
    operation:int = input("Operation no. :")
    if operation == '1':
        Data = input("String to encode:")
        print(Encode(Data))
        main()
    elif operation == '2':
        Data = input("Vectors to decode:")
        print(Decode(Data))
        main()
    elif operation == '3':
        config = {
            "embedding_size": 32,
            "number_of_epochs": int(input("target epochs:")),
            "batch_size":128,
            "learning_rate":0.01
        }

        train.Train(config=config)
        main()
    elif operation == '4':
        return 0
    else:
        print("Inalid operation.")
        return 1

if __name__ == "__main__":
    print(
        "Enter your desired operation:" \
        "1 = Decode" \
        "2 = Encode" \
        "3 = Train " \
        "4 = Exit"
    )
    main()