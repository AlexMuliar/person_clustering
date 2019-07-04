import pymysql
# import mysql.connector 

connection = pymysql.connect(
    host='127.0.0.1',
    user='root',
    db='molengo',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor  
)
# conn = mysql.connector.connect(
#     host='127.0.0.1',
#     user='root',
#     db='molengo',
#     charset='utf8mb4',
# )