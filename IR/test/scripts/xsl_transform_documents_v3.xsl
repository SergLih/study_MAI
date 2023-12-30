<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output method="xml" indent="yes" omit-xml-declaration="yes" xml:space="preserve"/>


  <xsl:template match="front">
  <Article>
    <Journal><xsl:value-of select="//journal-title"/></Journal>
    <Title><xsl:value-of select="//article-title"/></Title>
    <Doi><xsl:value-of select="//article-id[@pub-id-type='doi']"/></Doi>
    <Authors>
          <xsl:for-each select="//contrib">
              <xsl:value-of select="concat(name/surname, ' ', name/given-names)"/>
              <xsl:if test="position() != last()">, </xsl:if>
         </xsl:for-each>
      </Authors>
    <Abstact><xsl:value-of select="//abstract/p"/></Abstact>
    <Body>
        <xsl:for-each select="//body/descendant::*/text()">
            <xsl:value-of select="."/>
            <xsl:text> </xsl:text>
        </xsl:for-each>
    </Body>
   </Article>
 </xsl:template>
  
      <xsl:template match="@*|node()">
        <xsl:copy>
            <xsl:apply-templates select="front" />
        </xsl:copy>
    </xsl:template>
  </xsl:stylesheet>