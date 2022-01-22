import threading
import time
import os
import time
import telepot
import shutil

WithoutMask_path =r'C:\Users\Asus\Desktop\OOP_FinalProject\output\WithoutMask\\'
WithMask_path =r'C:\Users\Asus\Desktop\OOP_FinalProject\output\WithMask\\'
BotToken = '5026727343:AAGHof_rWhLjKS32SV3fUo_0LY6kZc5z2eo'

preID =""
file_chat_id = open('chat_id.txt','r')
Line = file_chat_id.readlines()
file_chat_id.close()
chat_id= str(Line[0])

# 子執行序 的工作內容
def find_file_name(id,tempurt):
    count = 0
    while True:
        # find in folder - WithoutMask
        list_of_fileName = os.listdir(WithoutMask_path)
        for file_name in list_of_fileName:
            file_nameID = str(file_name).split("_")
            if file_nameID[0] == id: # 有找到相同檔名
                fileName = file_name
                print("%s has been found in  WithoutMask  folder." % id)
                send_telegramBot("WithoutMask",fileName,float(tempurt))
                return

        # find in folder - WithMask
        list_of_fileName = os.listdir(WithMask_path)
        fileName =""        
        for file_name in list_of_fileName:
            file_nameID = str(file_name).split("_")
            if file_nameID[0] == id: # 有找到相同檔名
                fileName = file_name                
                print("%s has been found in  WithMask  folder." % id)
                send_telegramBot("WithMask",fileName,float(tempurt))
                return

        

        # close the thread after 20 sec        
        if count == 20:
            return
        time.sleep(1.0 - (time.time() - starttime) % 1.0)
        count+=1
        print("finding... %s_###.jpg" % id)

 
def create_thread(IDAndTpt):
    #idAndTpt = (EplyId,BodyTmptr)
    new_thread = threading.Thread(target = find_file_name,args = IDAndTpt)
    new_thread.start()


def send_telegramBot(path,fileName,tmptr):
    strTmptr =""
    eplyId = str(fileName).split("_")
    strcut_time = time.localtime(time.time())
    time_send = time.strftime("%Y-%m-%d- %H:%M",strcut_time)
    bot = telepot.Bot(BotToken)

    if tmptr >= 38.0:
        strTmptr = "體溫異常! "
    else:
        strTmptr = "體溫正常 "
    
    if path == "WithoutMask":
        try:
            open('output/WithoutMask/'+fileName, 'rb')
            bot.sendPhoto(chat_id, photo = open('output/WithoutMask/'+fileName, 'rb'),caption= time_send+"\n[入口A] \n員工 "+eplyId[0]+" 【沒戴口罩】!\n" + strTmptr + BodyTmptr[1])
            # new_loc = shutil.move(WithoutMask_path + fileName   , r"C:\Users\Asus\Desktop\OOP_FinalProject\output\sent")
            print("Bot sent ! %s" % fileName)
        except:
            print("[error ]no such file can send")
        
    elif path == "WithMask":
        if tmptr >=38 :
            try:
                bot.sendPhoto(chat_id, photo = open('output/WithMask/'+fileName, 'rb'),caption= time_send+"\n[入口A] \n員工 "+eplyId[0]+" 【有戴口罩】\n" + strTmptr +BodyTmptr[1])
                # new_loc = shutil.move(WithMask_path + fileName  , r"C:\Users\Asus\Desktop\OOP_FinalProject\output\sent")
                print("Bot sent ! %s" % fileName)
            except:
                print("[error ]no such file can send")
            
    


if __name__ == '__main__':
    starttime = time.time()
    while True:        
        file_Temperature = open('input/EmployeeInfo.txt','r')
        Lines = file_Temperature.readlines()
        file_Temperature.close()
        if Lines == []:
            continue
        else:            
            IdandTmptr = str(Lines[0]).split(",")
            EplyId = IdandTmptr[0].split(": ")
            BodyTmptr = IdandTmptr[1].split(": ")
            if EplyId[1] == preID:
                continue
        preID = str(EplyId[1])
        print("file_update")
        idAndTpt = (EplyId[1],BodyTmptr[1])
        create_thread(idAndTpt)



        time.sleep(2.0 - (time.time() - starttime) % 2.0)
        print('waiting capture...')