/*

   Derby - Class org.apache.derbyTesting.functionTests.tests.derbynet.badConnection

   Copyright 2002, 2004 The Apache Software Foundation or its licensors, as applicable.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

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

import java.io.BufferedOutputStream;
import org.apache.derbyTesting.functionTests.harness.TimedProcess;
import org.apache.derbyTesting.functionTests.util.TestUtil;

/**
	This tests various bad connection states
		- non-existant database
*/

public class badConnection
{
	
	private static Properties properties = new java.util.Properties();

	private static String dbNotFoundDB = "notthere";
	private static String  invalidAttrDB = "testbase;upgrade=notValidValue";
	private static String  derbynetDB = "testbase";


	private static Connection newConn(String database,Properties properties) throws Exception
	{
		Connection conn = null;
		String databaseURL = TestUtil.getJdbcUrlPrefix() + database;
		//System.out.println("URL is: " + databaseURL);

		try {
			conn = DriverManager.getConnection(databaseURL, properties); 
			if (conn == null)
				System.out.println("create connection didn't work");
			else
				System.out.println("Connection made\n");

		}
		catch (SQLException se)
		{
			showSQLException(se);
		}

		return conn;
	}

	private static void showSQLException(SQLException e)
	{
		System.out.println("passed SQLException all the way to client, then thrown by client...");
		System.out.println("SQLState is: "+e.getSQLState());
		System.out.println("vendorCode is: "+e.getErrorCode());
		System.out.println("nextException is: "+e.getNextException());
		System.out.println("reason is: "+e.getMessage() +"\n\n");
	}

	public static void main (String args[]) throws Exception
	{
		
		try
		{
			TestUtil.loadDriver();
			System.out.println("No user/password  (Client error)");
			Connection conn1 = newConn(derbynetDB, properties);

			System.out.println("Database not Found  (RDBNFNRM)");
			properties.put ("user", "admin");
			properties.put ("password", "admin");
			conn1 = newConn(dbNotFoundDB, properties);
			if (conn1 != null)
				conn1.close();

			System.out.println("Invalid Attribute  value (RDBAFLRM)");
			conn1 = newConn(invalidAttrDB, properties);
			if (conn1 != null)
				conn1.close();
		
		}
		catch (SQLException se)
		{
			showSQLException(se);
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}


}
