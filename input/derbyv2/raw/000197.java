package org.apache.derbyTesting.functionTests.tests.jdbcapi;


import java.sql.*;

import org.apache.derby.tools.ij;
import org.apache.derby.tools.JDBCDisplayUtil;

public class rsgetXXXcolumnNames {

    public static void main(String[] args) {
        test1(args);
    }
    
        public static void test1(String []args) {   
                Connection con;
                ResultSet rs;
                Statement stmt = null;
                PreparedStatement stmt1 = null;

                System.out.println("Test rsgetXXXcolumnNames starting");

                try
                {
                        // use the ij utility to read the property file and
                        // make the initial connection.
                        ij.getPropertyArg(args);
                        con = ij.startJBMS();
					
			con.setAutoCommit(false);                        			              

                        stmt = con.createStatement(); 

			// create a table with two columns, their names differ in they being in different cases.
                        stmt.executeUpdate("create table caseiscol(COL1 int ,\"col1\" int)");

   			con.commit();
   			
			stmt.executeUpdate("insert into caseiscol values (1,346)");

			con.commit();

                        // select data from this table for updating
			stmt1 = con.prepareStatement("select COL1, \"col1\" from caseiscol FOR UPDATE",ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_UPDATABLE);
		        rs = stmt1.executeQuery();

			// Get the data and disply it before updating.
                        System.out.println("Before updation...");
			while(rs.next()) {
			   System.out.println("ResultSet is: "+rs.getObject(1));
			   System.out.println("ResultSet is: "+rs.getObject(2));
			}
                        rs.close();
			rs = stmt1.executeQuery();
			while(rs.next()) {
			   // Update the two columns with different data.
			   // Since update is case insensitive only the first column should get updated in both cases.
			   rs.updateInt("col1",100);
			   rs.updateInt("COL1",900);
			   rs.updateRow();
			}
			rs.close();

			System.out.println("After update...");
			rs = stmt1.executeQuery();

			// Display the data after updating. Only the first column should have the updated value.
			while(rs.next()) {
			   System.out.println("Column Number 1: "+rs.getInt(1));
			   System.out.println("Column Number 2: "+rs.getInt(2));
			}
			rs.close();
			rs = stmt1.executeQuery();
			while(rs.next()) {
			   // Again checking for case insensitive behaviour here, should display the data in the first column.
			   System.out.println("Col COL1: "+rs.getInt("COL1"));
			   System.out.println("Col col1: "+rs.getInt("col1"));
			}
			rs.close();
 		} catch(SQLException sqle) {
 		   dumpSQLExceptions(sqle);
 		   sqle.printStackTrace();
 		} catch(Throwable e) {
 		   System.out.println("FAIL -- unexpected exception: "+e.getMessage());
                   e.printStackTrace();

 		}
     }
     
     static private void dumpSQLExceptions (SQLException se) {
                System.out.println("FAIL -- unexpected exception");
                while (se != null) {
                        System.out.println("SQLSTATE("+se.getSQLState()+"): "+se.getMessage());
                        se = se.getNextException();
                }
        }
}
