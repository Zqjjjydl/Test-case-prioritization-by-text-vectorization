/*

   Derby - Class org.apache.derbyTesting.functionTests.tests.jdbcapi.resultsetJdbc30

   Copyright 2001, 2005 The Apache Software Foundation or its licensors, as applicable.

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

package org.apache.derbyTesting.functionTests.tests.jdbcapi;

import java.sql.Array;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

import org.apache.derby.tools.ij;

/**
 * Test of additional methods in JDBC3.0 result set
 *
 */

public class resultsetJdbc30 { 
	public static void main(String[] args) {
		Connection con;
		ResultSet rs;
		Statement stmt;
		String[]  columnNames = {"i", "s", "r", "d", "dt", "t", "ts", "c", "v", "tn", "dc"};

		System.out.println("Test resultsetJdbc30 starting");

		try
		{
			// use the ij utility to read the property file and
			// make the initial connection.
			ij.getPropertyArg(args);
			con = ij.startJBMS();

			stmt = con.createStatement();

      //create a table, insert a row, do a select from the table,
			stmt.execute("create table t (i int, s smallint, r real, "+
				"d double precision, dt date, t time, ts timestamp, "+
				"c char(10), v varchar(40) not null, dc dec(10,2))");
			stmt.execute("insert into t values(1,2,3.3,4.4,date('1990-05-05'),"+
						 "time('12:06:06'),timestamp('1990-07-07 07:07:07.07'),"+
						 "'eight','nine', 11.1)");

			rs = stmt.executeQuery("select * from t");
			rs.next();

			//following will give not implemented exceptions.
			try {
			  System.out.println();
			  System.out.println("trying rs.getURL(int) :");
			  rs.getURL(8);
			  System.out.println("Shouldn't reach here. Method not implemented yet.");
 			} catch (SQLException ex) {
			  System.out.println("Expected : " + ex.getMessage());
 			}

			try {
			  System.out.println();
			  System.out.println("trying rs.getURL(String) :");
			  rs.getURL("c");
			  System.out.println("Shouldn't reach here. Method not implemented yet.");
 			} catch (SQLException ex) {
			  System.out.println("Expected : " + ex.getMessage());
 			}

			try {
			  System.out.println();
			  System.out.println("trying rs.updateRef(int, Ref) :");
			  rs.updateRef(8,null);
			  System.out.println("Shouldn't reach here. Method not implemented yet.");
 			} catch (SQLException ex) {
			  System.out.println("Expected : " + ex.getMessage());
 			}

			try {
			  System.out.println();
			  System.out.println("trying rs.updateRef(String, Ref) :");
			  rs.updateRef("c",null);
			  System.out.println("Shouldn't reach here. Method not implemented yet.");
 			} catch (SQLException ex) {
			  System.out.println("Expected : " + ex.getMessage());
 			}

			try {
			  System.out.println();
			  System.out.println("trying rs.updateBlob(int, Blob) :");
			  rs.updateBlob(8,null);
			  System.out.println("Shouldn't reach here because method is being invoked on a read only resultset");
 			} catch (SQLException ex) {
			  System.out.println("Expected : " + ex.getMessage());
 			}

			try {
			  System.out.println();
			  System.out.println("trying rs.updateBlob(String, Blob) :");
			  rs.updateBlob("c",null);
			  System.out.println("Shouldn't reach here because method is being invoked on a read only resultset");
 			} catch (SQLException ex) {
			  System.out.println("Expected : " + ex.getMessage());
 			}

			try {
			  System.out.println();
			  System.out.println("trying rs.updateClob(int, Clob) :");
			  rs.updateClob(8,null);
			  System.out.println("Shouldn't reach here because method is being invoked on a read only resultset");
 			} catch (SQLException ex) {
			  System.out.println("Expected : " + ex.getMessage());
 			}

			try {
			  System.out.println();
			  System.out.println("trying rs.updateClob(String, Clob) :");
			  rs.updateClob("c",null);
			  System.out.println("Shouldn't reach here because method is being invoked on a read only resultset");
 			} catch (SQLException ex) {
			  System.out.println("Expected : " + ex.getMessage());
 			}

			try {
			  System.out.println();
			  System.out.println("trying rs.updateArray(int, Array) :");
			  rs.updateArray(8,null);
			  System.out.println("Shouldn't reach here. Method not implemented yet.");
 			} catch (SQLException ex) {
			  System.out.println("Expected : " + ex.getMessage());
 			}

			try {
			  System.out.println();
			  System.out.println("trying rs.updateClob(String, Array) :");
			  rs.updateArray("c",null);
			  System.out.println("Shouldn't reach here. Method not implemented yet.");
 			} catch (SQLException ex) {
			  System.out.println("Expected : " + ex.getMessage());
 			}

			rs.close();

			stmt.close();
			con.close();

		}
		catch (SQLException e) {
			dumpSQLExceptions(e);
			e.printStackTrace();
		}
		catch (Throwable e) {
			System.out.println("FAIL -- unexpected exception: "+e);
			e.printStackTrace();
		}

		System.out.println("Test resultsetJdbc30 finished");
    }

	static private void dumpSQLExceptions (SQLException se) {
		System.out.println("FAIL -- unexpected exception");
		while (se != null) {
			System.out.println("SQLSTATE("+se.getSQLState()+"): "+se);
			se = se.getNextException();
		}
	}

}
