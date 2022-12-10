package org.apache.derbyTesting.functionTests.tests.jdbcapi;


import java.sql.*;

import org.apache.derby.tools.ij;
import org.apache.derby.tools.JDBCDisplayUtil;

public class testRelative {
   
   static final String NO_CURRENT_ROW_SQL_STATE = "24000";
  
   public static void main(String[] args) {
	   System.out.println("Test testRelative starting");
	   Connection con = null;
	   try {
		   // use the ij utility to read the property file and
		   // make the initial connection.
		   ij.getPropertyArg(args);
		   con = ij.startJBMS();
		   test1(con);        
	   } catch (Exception e)
	   {
		   unexpectedException(e);
	   }
    }
    
    public static void test1(Connection con) {
		ResultSet rs = null;
		PreparedStatement pStmt = null;
		Statement stmt1 = null;
		String returnValue = null;
		
		try
		{
			con.setAutoCommit(false);
			pStmt = con.prepareStatement("create table testRelative(name varchar(10), i int)");
   			pStmt.executeUpdate();
   			con.commit();
   			
   			pStmt = con.prepareStatement("insert into testRelative values (?,?)");
   			   			
   			pStmt.setString(1,"work1");
			pStmt.setNull(2,1);
			pStmt.addBatch();
			
			pStmt.setString(1,"work2");
			pStmt.setNull(2,2);
			pStmt.addBatch();
			
			pStmt.setString(1,"work3");
			pStmt.setNull(2,3);
			pStmt.addBatch();
			
			pStmt.setString(1,"work4");
			pStmt.setNull(2,4);
			pStmt.addBatch();

		
			pStmt.executeBatch();
			con.commit();

			stmt1 = con.createStatement(ResultSet.TYPE_SCROLL_INSENSITIVE,ResultSet.CONCUR_READ_ONLY);
		        rs = stmt1.executeQuery("select * from testRelative");						

   			rs.next(); // First Record
   			returnValue = rs.getString("name");
   			System.out.println("Value="+returnValue);

   			rs.relative(2);
   			System.out.println("isFirst=" + rs.isFirst() + " isLast=" + rs.isLast() + " isAfterLast=" + rs.isAfterLast());
   			returnValue = rs.getString("name");
   			System.out.println("Value="+returnValue);

   			rs.relative(-2);
   			returnValue = rs.getString("name");
   			System.out.println("Value="+returnValue);
                 } catch(SQLException se) {
		    unexpectedSQLException(se);
                 } catch(Throwable t) {
		    System.out.println("FAIL--unexpected exception: "+t.getMessage());
		    t.printStackTrace(System.out);
                 }

                 try {

   			rs.relative(10);
   			System.out.println("isFirst=" + rs.isFirst() + " isLast=" + rs.isLast() + " isAfterLast=" + rs.isAfterLast());

   			returnValue = rs.getString("name");
   			System.out.println("Value="+returnValue);

 		} catch(SQLException sqle) {
 		   expectedException(sqle, NO_CURRENT_ROW_SQL_STATE);
 		} catch(Throwable e) {
 		   System.out.println("FAIL -- unexpected exception: "+e.getMessage());
                   e.printStackTrace(System.out);

 		}
      }
     
      /**
	   *  Print the expected Exception's details if the SQLException SQLState
	   * matches the expected SQLState. Otherwise fail
	   *
	   * @param se  SQLException that was thrown by the test
	   * @param expectedSQLState  The SQLState that we expect. 
	   *
	   **/
	static private void expectedException (SQLException se, String expectedSQLState) {
           if( se.getSQLState() != null && (se.getSQLState().equals(expectedSQLState))) { 
                System.out.println("PASS -- expected exception");
                while (se != null) {
                    System.out.println("SQLSTATE("+se.getSQLState()+"): "+se.getMessage());
                    se = se.getNextException();
                }
            } else {
	        System.out.println("FAIL--Unexpected SQLException: " +
							   "SQLSTATE(" +se.getSQLState() + ")" +
							   se.getMessage());
	        se.printStackTrace(System.out);
	    }
	}

     /**
       * We are here because we got an exception when did not expect one.
       * Hence printing the message and stack trace here.
       **/
     static private void unexpectedSQLException(SQLException se) {
	 System.out.println("FAIL -- Unexpected Exception: "+ 
						"SQLSTATE(" +se.getSQLState() +")" +
						se.getMessage());
	 se.printStackTrace(System.out);
     }

	static private void unexpectedException(Exception e) {
		System.out.println("FAIL -- Unexpected Exception: "+ 
						   e.getMessage());
	}
	
}
