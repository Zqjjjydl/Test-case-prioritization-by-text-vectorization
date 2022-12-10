/*

   Derby - Class org.apache.derbyTesting.functionTests.tests.derbynet.testij

   Licensed to the Apache Software Foundation (ASF) under one or more
   contributor license agreements.  See the NOTICE file distributed with
   this work for additional information regarding copyright ownership.
   The ASF licenses this file to You under the Apache License, Version 2.0
   (the "License"); you may not use this file except in compliance with
   the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

 */
package org.apache.derbyTesting.functionTests.tests.derbynet;

import java.sql.*;
import java.util.Vector;
import java.util.Properties;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.BufferedOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.ByteArrayInputStream;
import java.io.FileNotFoundException;

import org.apache.derbyTesting.functionTests.harness.jvm;
import org.apache.derbyTesting.functionTests.harness.Sed;
import org.apache.derbyTesting.functionTests.util.TestUtil;
import org.apache.derbyTesting.functionTests.util.ExecProcUtil;

import org.apache.derby.drda.NetworkServerControl;


public class testij
{


	private static Properties properties = new java.util.Properties();
	private static jvm jvm;
	private static Vector vCmd;

	private static String IjCmd="org.apache.derby.tools.ij";
	private static String SqlDir="extin";
	private static String jccSqlFile="testij.sql";
	private static String sep;
	private static String clientSqlFile="testclientij.sql";
	private static String altExtinDir;
	private static boolean useAltExtinDir=false;
    
	private static void execCmd (String[] args) throws Exception
	{
		int totalSize = vCmd.size() + args.length;
		String serverCmd[] = new String[totalSize];
		int i;
		for (i = 0; i < vCmd.size(); i++)
		{
			serverCmd[i] = (String)vCmd.elementAt(i);
		//	System.out.println("serverCmd["+i+"]: "+serverCmd[i]);
		}
		int j = 0;
		for (; i < totalSize; i++)
		{
			serverCmd[i] = args[j++];
		//	System.out.println("serverCmd["+i+"]: "+serverCmd[i]);
		}
 
		// Start a process to run the command
		Process pr = Runtime.getRuntime().exec(serverCmd);
		pr.waitFor();		// make sure this is executed first
	}
	
    public static void massageSqlFile (String hostName, String fileName) throws Exception {
        // only called if hostName is *not* localhost. 
        // Need to replace each occurrence of the string 'localhost' with 
        // whatever is the hostName
        File tmpFile = new File("extin", "tmpFile.sql");
        File orgFile = new File("extin", fileName);
        // wrap this in a try to possibly try using user.dir to find the file
        InputStream original; 
        OutputStream copy; 
        try { 
            fileName = SqlDir + sep + fileName; 
            original = new FileInputStream(fileName);
            copy = new FileOutputStream(tmpFile);
        }
        catch (FileNotFoundException fnfe) {
            // we must be running from within a suite...
            useAltExtinDir = true;
            String userdir =  System.getProperty("user.dir");
            altExtinDir = userdir + sep + ".."; 
            tmpFile = new File(altExtinDir, "tmpFile.sql");
            orgFile = new File (altExtinDir,  fileName); 
            fileName = altExtinDir + sep + fileName;
            original = new FileInputStream(fileName);
            copy = new FileOutputStream(tmpFile);
        }
        int content;
        while ((content = original.read())> 0 ) {
            copy.write(content);
        }
        copy.close();
        original.close();
        Sed hostSed = new Sed();
        InputStream sedIs = new ByteArrayInputStream(("substitute=localhost;" + hostName).getBytes("UTF-8"));
        hostSed.exec(tmpFile, orgFile, sedIs, false, false, false);		
    }

	public static void main (String args[]) throws Exception
	{
		if ((System.getProperty("java.vm.name") != null) && System.getProperty("java.vm.name").equals("J9"))
			jvm = jvm.getJvm("j9_13");
		else
			jvm = jvm.getJvm("currentjvm");		// ensure compatibility
		vCmd = jvm.getCommandLine();
		sep =  System.getProperty("file.separator");
		try
		{
            BufferedOutputStream bos = new BufferedOutputStream(System.out, 1024);
			/************************************************************
			 *  Test comments in front of select's doesn't cause problems
			 ************************************************************/
			//create wombat database
			NetworkServerControl server = new NetworkServerControl();
			System.out.println("Testing various ij connections and comments in front of selects");
			
			// first, we have to massage the .sql file to replace localhost, if 
			// there is a system property set.
						
			String hostName=TestUtil.getHostName();
			if (TestUtil.isJCCFramework()){
				// use jccSqlfile
				if (!hostName.equals("localhost")) 
					massageSqlFile(hostName,jccSqlFile);
				if (useAltExtinDir)	
					ExecProcUtil.execCmdDumpResults(new String[]{IjCmd,(altExtinDir + sep + SqlDir + sep + jccSqlFile)},vCmd,bos);
				ExecProcUtil.execCmdDumpResults(new String[]{IjCmd,(SqlDir + sep + jccSqlFile)},vCmd,bos);
			} else {   // Derby Client
				// use clientSqlFile
				if(!hostName.equals("localhost")) {
					massageSqlFile(hostName,clientSqlFile);
				if (useAltExtinDir)	
					ExecProcUtil.execCmdDumpResults(new String[]{IjCmd,(altExtinDir + sep + SqlDir + sep + clientSqlFile)},vCmd,bos);
				}
				ExecProcUtil.execCmdDumpResults(new String[]{IjCmd,(SqlDir + sep + clientSqlFile)},vCmd,bos);
			}
			System.out.println("End test");
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}

}


