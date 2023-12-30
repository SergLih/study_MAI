<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output method="xml" indent="yes" omit-xml-declaration="yes" encoding="utf-8"
              cdata-section-elements="Availability"/>
              

<xsl:template match="geoData">
    <Coordinates>
        <xsl:attribute name="N">
             <xsl:value-of select="coordinates/array[1]"/>
        </xsl:attribute>
        <xsl:attribute name="E">
            <xsl:value-of select="coordinates/array[2]"/>
        </xsl:attribute>
    </Coordinates>
  </xsl:template>
  
  <xsl:template match="/catalog">
    <SportsComplexes>
      <xsl:for-each select="/catalog/array">
         <SportsComplex>
            <WorkingHours>
              <xsl:apply-templates select="WorkingHours" />
            </WorkingHours>
              <xsl:apply-templates select="*[not(name() = 'WorkingHours')]"/> 
         </SportsComplex>
      </xsl:for-each>
     </SportsComplexes>
  </xsl:template>
  
  <xsl:template match="WorkingHours">
      <Day>
      <xsl:attribute name="name">
         <xsl:value-of select="DayWeek"/>
      </xsl:attribute>
      <xsl:attribute name="open">  
        <xsl:value-of select="substring-before(WorkHours, '-')" />
      </xsl:attribute>
      <xsl:attribute name="close">  
        <xsl:value-of select="substring-after(WorkHours, '-')" />
      </xsl:attribute>
      </Day>
  </xsl:template>
  
  <xsl:template match="ObjectAddress">
    <Address>
        <xsl:attribute name="postal_code">  
            <xsl:value-of select="PostalCode" />
        </xsl:attribute>
        <xsl:attribute name="city">  
            <xsl:value-of select="substring-before(Address, ',')" />
        </xsl:attribute>
        <xsl:attribute name="address">  
            <xsl:value-of select="substring-after(Address, ',')" />
        </xsl:attribute>
        <xsl:attribute name="adm_area">  
            <xsl:value-of select="AdmArea" />
        </xsl:attribute>
        <xsl:attribute name="district">  
            <xsl:value-of select="District" />
        </xsl:attribute>
    </Address>
     <xsl:apply-templates select="Availability" />
  </xsl:template>
  
  <xsl:template match="Availability">
    <Availability>
    <xsl:for-each select="available_element">
        <xsl:sort select="Area_mgn" />
        <xsl:sort select="Element_mgn" />
        
        <xsl:value-of select="Area_mgn"/>:&#09; <xsl:value-of select="Element_mgn"/>&#09; [<xsl:value-of select="Group_mgn"/>]&#09;  <xsl:value-of select="available_index"/> (<xsl:value-of select="available_degree"/>)
    </xsl:for-each>
      </Availability>
  </xsl:template>
  
    <xsl:template match="@*|node()">
        <xsl:copy>
            <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
    </xsl:template>
  </xsl:stylesheet>