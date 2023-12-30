<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:fn="http://www.w3.org/2005/xpath-functions">
<xsl:output method="xml" version="1.0" encoding="UTF-8" indent="yes"/>
<xsl:template name="main">
<Articles>
   <xsl:apply-templates select="collection('/home/sergey/MAI/maga/Inf_search/test?select=*.nxml;recurse=yes')"/>
</Articles>
</xsl:template>

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
   </Article>
 </xsl:template>
  
      <xsl:template match="@*|node()">
        <xsl:copy>
            <xsl:apply-templates select="front" />
        </xsl:copy>
    </xsl:template>
  </xsl:stylesheet>
