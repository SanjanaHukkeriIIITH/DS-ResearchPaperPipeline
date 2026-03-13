import os
import sys

def check_env():
    print("--- Spark Windows Diagnostic Tool ---")
    print(f"OS: {os.name}")
    print(f"Python Version: {sys.version}")
    
    hadoop_home = os.environ.get("HADOOP_HOME")
    spark_home = os.environ.get("SPARK_HOME")
    java_home = os.environ.get("JAVA_HOME")
    
    print(f"HADOOP_HOME: {hadoop_home}")
    print(f"SPARK_HOME: {spark_home}")
    print(f"JAVA_HOME: {java_home}")
    
    if os.name == 'nt':
        local_hadoop = os.path.abspath("hadoop")
        if not hadoop_home and os.path.exists(os.path.join(local_hadoop, "bin", "winutils.exe")):
            print(f"[OK] Found local Hadoop fallback at {local_hadoop}")
            os.environ["HADOOP_HOME"] = local_hadoop
            hadoop_home = local_hadoop
            
        if hadoop_home:
            hadoop_bin = os.path.join(hadoop_home, "bin")
            if hadoop_bin not in os.environ["PATH"]:
                os.environ["PATH"] = hadoop_bin + os.path.pathsep + os.environ["PATH"]
                print(f"[OK] Added {hadoop_bin} to PATH")
        else:
            print("\n[ERROR] HADOOP_HOME is not set.")
        
        if hadoop_home:
            winutils_path = os.path.join(hadoop_home, "bin", "winutils.exe")
            hadoop_dll_path = os.path.join(hadoop_home, "bin", "hadoop.dll")
            
            if os.path.exists(winutils_path):
                print(f"[OK] winutils.exe found at {winutils_path}")
            else:
                print(f"[ERROR] winutils.exe NOT found at {winutils_path}")
                
            if os.path.exists(hadoop_dll_path):
                print(f"[OK] hadoop.dll found at {hadoop_dll_path}")
            else:
                print(f"[WARNING] hadoop.dll NOT found at {hadoop_dll_path}. It might be needed for some operations.")

    print("\n--- Testing Spark Initialization ---")
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName("Diagnostic").master("local[1]").getOrCreate()
        print("[OK] SparkSession created successfully!")
        spark.stop()
    except Exception as e:
        print(f"[FAIL] Spark initialization failed: {e}")

if __name__ == "__main__":
    check_env()
