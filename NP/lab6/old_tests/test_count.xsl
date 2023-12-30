<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output method="text" />
  <xsl:key name="elements" match="//@*" use="local-name()" />
  <xsl:template match="/">
    <xsl:value-of select="count(//@*[count(.|key('elements', local-name())[1]) = 1])"/>
    <xsl:for-each select="//@*[count(.|key('elements', local-name())[1]) = 1]">
      <xsl:value-of select="local-name()" />, 
    </xsl:for-each>
  </xsl:template>
</xsl:stylesheet>
