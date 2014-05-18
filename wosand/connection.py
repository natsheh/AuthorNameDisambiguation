import mysql.connector
from mysql.connector import errorcode

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': '3306',
    'user': '',
    'password': '',
    'database': '',
    'raise_on_warnings': True,
}

class Connection:
    cnx = None
    
    def __init__(self):
        try:
            self.cnx = mysql.connector.connect(**DB_CONFIG)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exists")
            else:
                print(err)
    
    def execute(self, query, mult=False):
        records = None
        if self.cnx and query:
            cursor = self.cnx.cursor()
            r = cursor.execute(query, multi=mult)
            if mult:
                for x in r:
                    if x.with_rows:
                        records = x
            else:
                records = cursor.fetchall()
        return records
        
    def close(self):
        self.cnx.close()
        
    def set_group_limit(self, limit):
        if self.cnx and limit:
            cursor = self.cnx.cursor()
            cursor.execute("SET LOCAL group_concat_max_len={0}".format(limit))