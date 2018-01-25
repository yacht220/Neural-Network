<logheader platform="Linux" env-locale-cs="ANSI_X3.4-1968" process_id="MOD-1510821025182" version="7.5.100-183" tz="EST" />

--------------------------------------------------------------------------------------------------------
-----  START MOD-1510821025182 log 2017-11-16 03:31:03 EST  [7.5.100-183 Linux-x86_64]
--------------------------------------------------------------------------------------------------------

2017-11-16 03:31:03 MOD-1510821025182 Info <0000>: Workorder : MOD-1510821025182
2017-11-16 03:31:03 MOD-1510821025182 Info <0000>: Plugin ID : 14003-Isilon Filer
2017-11-16 03:31:03 MOD-1510821025182 Info <0000>: bindir    : /usr/local/avamar/bin
2017-11-16 03:31:03 MOD-1510821025182 Info <0000>: vardir    : /usr/local/avamar/var/clidev-isilon2-2-vm-griffin-6173
2017-11-16 03:31:03 MOD-1510821025182 Info <0000>: Initiator : 
2017-11-16 03:31:03 MOD-1510821025182 Info <0000>: Order      : 
<snapup pid="Isilon Filer" time="1510821045" did="on-demand" ack="false" type="work" msgver="5" pidnum="14003" work="backup" ticket="XXX" key="14003-MOD-1510821025182" sync="bg" sessionid="613d362e5b2574d4638c45b36c454b2f5c53f109" cid="0a4fc2dcc3ff974d81f16ce9c595fa5162d0d1b5" mid="3844/3863/39" gsansid="9151082102518209" last_ctl_man_key="14003-MOD-1510821025182" wid="MOD-1510821025182" customaction="">
  <targetlist>
    <path backup="true" name="/ifs/100MB_br/">
      <dir name="100MB_br" size="915" selectable="true" fstype="container" />
    </path>
  </targetlist>
  <directives>
    <flag type="string" value="tls" name="encrypt" />
    <flag type="string" value="high" name="encrypt-strength" />
    <flag type="string" value="1516005045" name="expire" />
    <flag type="string" value="none" name="retention-type" />
    <flag type="string" value="vm-griffin-6177.asl.lab.emc.com" name="hfsaddr" />
    <flag type="string" value="27000" name="hfsport" />
    <flag type="password" value="****" name="ddr-auth-token-key" />
    <flag type="string" value="backuponly" name="id" />
    <flag type="password" value="****" name="ap" />
    <flag type="string" value="/yizhou/clidev-isilon2-2-vm-griffin-6173" name="path" />
  </directives>
  <catalog>
    <flag type="pulldown" value="default" name="[avtar]ddr-encrypt-strength" />
    <flag type="checkbox" value="true" name="ddr" />
    <flag type="pulldown" value="1" name="ddr-index" />
    <flag type="pulldown" pidnum="14003" value="4" name="max-streams" />
    <flag type="pulldown" pidnum="14003" value="try-incremental" name="mode" />
  </catalog>
  <browse-from-catalog>
    <flag type="password" pidnum="14003" value="****" name="filerbrowsepswd" />
    <flag type="string" pidnum="14003" value="root" name="filerbrowseuser" />
  </browse-from-catalog>
  <free-form />
</snapup>

2017-11-16 03:31:03 MOD-1510821025182 Info <0000>: Executable : /usr/local/avamar/bin/avndmp
2017-11-16 03:31:03 MOD-1510821025182 Info <0000>: Args       : avndmp --sysdir="/usr/local/avamar/etc/clidev-isilon2-2-vm-griffin-6173" --bindir="/usr/local/avamar/bin" --vardir="/usr/local/avamar/var/clidev-isilon2-2-vm-griffin-6173" --ctlcallport="41737" --ctlinterface="14003-MOD-1510821025182" --logtimeformat="default" --logtimezone="default" --logfile="/usr/local/avamar/var/clidev-isilon2-2-vm-griffin-6173/MOD-1510821025182-14003-Isilon_Filer.log"
<SUMMARY totalprotectedbytes="556222976" filebytes_ddsent="275837" filecount_excluded="0" dirbytes_sent="21077" filebytes_sent="0" compbytes_ispresent="0" filebytes_ispresent="0" dirbytes_hashcache="8676" filebytes_skipped="0" totalprimarybytes="555942876" compbytes_sent="0" dirbytes_ispresent="0" chunkcount_sent="0" chunkcount_hashcache="15" filebytes_hardlink="0" dircount_sent="10" filebytes_noreason="0" filebytes_reduced="21644" filecount_hardlink="0" filecount_noreason="0" filecount_filecache="0" filebytes_ddpresent="555986304" filecount_skipped="0" compcount_hashcache="0" dircount_hashcache="6" snapview_stat="false" filebytes_filecache="0" compcount_sent="0" dircount_ispresent="0" chunkcount_ispresent="0" elapsed="37" compbytes_hashcache="0" filebytes_hashcache="9076" filecount_reduced="15" compcount_ispresent="0" totalfilebytes="556222976" totalfilecount="109" filebytes_excluded="0"><session numthreads="1" type="avtarbackup" expires="1516005045" ndispatchers="1" domain="" workorderid="MOD-1510821025182" numconns="1" pidnum="14003" label="MOD-1510821025182" path="/yizhou/clidev-isilon2-2-vm-griffin-6173" encrypt="tls" starttime="1510821104" labelnum="5" dispatcher0="10.25.106.177:29000" sessionid="9151082102518209" root="/" pluginid="Isilon Filer" encrypt-strength="high" user="" clientid="0a4fc2dcc3ff974d81f16ce9c595fa5162d0d1b5"><host numprocs="1" speed="200" osuser="root" name="vm-griffin-6173" memory="934" /><build msgversion="13-10" time="11:47:51" appname="avtar" zlibversion="1.2.3" lzoversion="1.08 Jul 12 2002" date="Jun 12 2017" appversion="7.5.100-183" processortype="x86_64" osversion="SLES-64" sslversion="TLSv1   OpenSSL 1.0.2a-fips 19 Mar 2015" osname="Linux" /><dirstats /></session><errorsummary exitcode="0" errors="0" warnings="0" fatals="0" /></SUMMARY>
<SUMMARY totalprotectedbytes="556222976" filebytes_ddsent="275837" filecount_excluded="0" dirbytes_sent="21077" filebytes_sent="0" compbytes_ispresent="0" filebytes_ispresent="0" dirbytes_hashcache="8676" filebytes_skipped="0" totalprimarybytes="555942876" compbytes_sent="0" dirbytes_ispresent="0" chunkcount_sent="0" chunkcount_hashcache="15" filebytes_hardlink="0" dircount_sent="10" filebytes_noreason="0" filebytes_reduced="21644" filecount_hardlink="0" filecount_noreason="0" filecount_filecache="0" filebytes_ddpresent="555986304" filecount_skipped="0" compcount_hashcache="0" dircount_hashcache="6" snapview_stat="false" filebytes_filecache="0" compcount_sent="0" dircount_ispresent="0" chunkcount_ispresent="0" elapsed="37" compbytes_hashcache="0" filebytes_hashcache="9076" filecount_reduced="15" compcount_ispresent="0" totalfilebytes="556222976" totalfilecount="109" filebytes_excluded="0">
  <session numthreads="1" type="avtarbackup" expires="1516005045" ndispatchers="1" domain="" workorderid="MOD-1510821025182" numconns="1" pidnum="14003" label="MOD-1510821025182" path="/yizhou/clidev-isilon2-2-vm-griffin-6173" encrypt="tls" starttime="1510821104" labelnum="5" dispatcher0="10.25.106.177:29000" sessionid="9151082102518209" root="/" pluginid="Isilon Filer" encrypt-strength="high" user="" clientid="0a4fc2dcc3ff974d81f16ce9c595fa5162d0d1b5">
    <host numprocs="1" speed="200" osuser="root" name="vm-griffin-6173" memory="934" />
    <build msgversion="13-10" time="11:47:51" appname="avtar" zlibversion="1.2.3" lzoversion="1.08 Jul 12 2002" date="Jun 12 2017" appversion="7.5.100-183" processortype="x86_64" osversion="SLES-64" sslversion="TLSv1   OpenSSL 1.0.2a-fips 19 Mar 2015" osname="Linux" />
    <dirstats />
  </session>
  <errorsummary exitcode="0" errors="0" warnings="0" fatals="0" />
</SUMMARY>

2017-11-16 03:31:50 MOD-1510821025182 Info <0000>: <workorder_finished pid="Isilon Filer" startUTC="2017-11-16 08:31:03" msgver="1" pidnum="14003" finishUTC="2017-11-16 08:31:50" errorcode="0" wid="MOD-1510821025182" errormessage="code 0: success" errorrealm="Plugin 14003-Isilon Filer" />

