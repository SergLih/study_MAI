<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output method="xml" indent="yes" omit-xml-declaration="yes"/>
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
  
  <xsl:template match="catalog/array">
      <WorkingHours>
          <xsl:for-each select="WorkingHours">
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
          </xsl:for-each>
      </WorkingHours>
      <xsl:apply-templates select="*[not(name() = 'WorkingHours')]" />
  </xsl:template>
  
  
  
    <xsl:template match="@*|node()">
        <xsl:copy>
            <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
    </xsl:template>
  </xsl:stylesheet>