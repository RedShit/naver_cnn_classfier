package com.thinkingmaze.test;

import lombok.AllArgsConstructor;  
import lombok.Data;  
import lombok.NoArgsConstructor;  
import lombok.extern.log4j.Log4j;  
  
@Data  
@NoArgsConstructor  
@AllArgsConstructor  
@Log4j  
public class LombokUser {  
      
    private String id = null;  
    private String name = null;  
    private String email = null;  
      
    public static void main(String[] args) {  
        
    }  
}  