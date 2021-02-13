import os
import sys
ip = sys.argv[1]
username = sys.argv[2]
passwd = sys.argv[3]
remote_dir = sys.argv[4]
acct_id = sys.argv[5]
os.system('sshpass -p '+passwd + ' scp -o StrictHostKeyChecking=no ' +'/root/accounts/account_'+acct_id+'/data_extracted/retraining_data.csv '+  username+'@'+ip+':'+remote_dir)
