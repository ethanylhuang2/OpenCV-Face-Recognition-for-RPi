from threading import Thread, Lock

t2 = Thread(target=print, args=("This is the second thread",))
t2.start()
while True:
    print("This is the main thread")