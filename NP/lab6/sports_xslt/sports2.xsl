<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:template match="/">
<html>
<head>
<script
  src="https://code.jquery.com/jquery-3.5.1.min.js"
  integrity="sha256-9/aliU8dGd2tb6OSsuzixeV4y/faTqgFtohetphbbj0="
  crossorigin="anonymous"></script>
<script src="http://cdn.datatables.net/1.10.22/js/jquery.dataTables.min.js"></script>
<link rel="stylesheet" type="text/css" href="http://cdn.datatables.net/1.10.22/css/jquery.dataTables.min.css" />
<style>
	.smallTable{
		width: 150px; 
		line-height: 1; 
		font-size: 70%;
	}
	.smallTable td {
		padding: 0 !important;
	}
	
</style>
<title>Спортивные объекты Москвы</title>
</head>
<body>
<table class="dataTable" id="example">
    <thead>
        <tr>
            <th>Название</th>
			<th>Категория</th>
            <th>Район</th>
            <th>Адрес</th>
			<th>Email</th>
			<th>Телефон</th>
			<th>Режим работы</th>
        </tr>
    </thead>
    <tbody>
        <xsl:for-each select="SportsComplexes/SportsComplex">
		<xsl:variable name="coordN" select="Coordinates/@N"/>
		<xsl:variable name="coordE" select="Coordinates/@E"/>
		<xsl:variable name="firstEmail" select="Email/Email[1]" />
		<xsl:variable name="firstPhone" select="PublicPhone/PublicPhone[1]" />
		<tr>
			<td><xsl:value-of select="ShortName"/></td>
			<td><xsl:value-of select="Category"/></td>
			<td><xsl:value-of select="Address/@district"/></td>
			<td><a target="_blank" href="https://maps.yandex.ru/?text={$coordE}+{$coordN}">
				<xsl:value-of select="Address/@address"/></a></td>
			<td><a target="_blank" href="mailto:{$firstEmail}"><xsl:value-of select="$firstEmail"/></a></td>
			<td><a target="_blank" href="tel:+7{$firstPhone}"><xsl:value-of select="$firstPhone"/></a></td>
			<td>
				<table class="smallTable">
				<xsl:for-each select="WorkingHours/Day">
					<xsl:variable name="weekdayLong" select="@name"/>
					<xsl:variable name="weekdayShort">
						<xsl:choose>
							<xsl:when test="$weekdayLong='понедельник'">пн</xsl:when>
							<xsl:when test="$weekdayLong='вторник'">вт</xsl:when>
							<xsl:when test="$weekdayLong='среда'">ср</xsl:when>
							<xsl:when test="$weekdayLong='четверг'">чт</xsl:when>
							<xsl:when test="$weekdayLong='пятница'">пт</xsl:when>
							<xsl:when test="$weekdayLong='суббота'">сб</xsl:when>
							<xsl:when test="$weekdayLong='воскресенье'">вс</xsl:when>
						</xsl:choose>
					</xsl:variable>
					<tr>
						<td><xsl:value-of select="$weekdayShort"/></td>
						<td><xsl:value-of select="@open"/>–
							<xsl:value-of select="@close"/></td>
					</tr>
				</xsl:for-each>
				</table>
			</td>
		</tr>
		</xsl:for-each>
    </tbody>
</table>
<script>
$(document).ready( function () {
    $('#example').DataTable();
} );
</script>
</body>
</html>
</xsl:template>
</xsl:stylesheet>